//! GHI Pipeline panel: orchestrates video -> frames (ffmpeg) -> COLMAP (CUDA)
//! -> Brush training, end to end from the GUI.
//!
//! The external stages run as child processes on a worker thread; progress is
//! streamed back to the panel via a channel and the panel kicks off training
//! through the regular `create_process` path once COLMAP finishes.

use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::sync::mpsc::{Receiver, Sender, channel};

use brush_process::{DataSource, create_process};

use crate::ui::panels::AppPane;
use crate::ui::ui_process::UiProcess;

#[derive(PartialEq, Clone, Copy)]
enum VideoMode {
    /// Equirectangular 360 video (Insta360): unwrap into 4 pinhole faces.
    Equirect360,
    /// Regular pinhole video: plain frame extraction.
    Pinhole,
}

enum PipeEvent {
    Log(String),
    Stage(usize),
    Finished(PathBuf),
    Failed(String),
}

const STAGES: [&str; 5] = [
    "Extraer frames (ffmpeg)",
    "COLMAP: features",
    "COLMAP: matching",
    "COLMAP: mapper",
    "Entrenamiento",
];

pub struct GhiPipelinePanel {
    video: String,
    workdir: String,
    ffmpeg: String,
    colmap: String,
    mode: VideoMode,
    fps: u32,
    train_steps: u32,
    running: bool,
    stage: usize,
    done_stages: usize,
    log: Vec<String>,
    rx: Option<Receiver<PipeEvent>>,
    pending_train: Option<PathBuf>,
    error: Option<String>,
}

impl Default for GhiPipelinePanel {
    fn default() -> Self {
        Self {
            video: String::new(),
            workdir: String::new(),
            ffmpeg: "ffmpeg".to_owned(),
            colmap: r"C:\Users\ilasso\OneDrive - GHI HORNOS INDUSTRIALES S.L\Escritorio\LIdar\Tools\cuda\bin\colmap.exe".to_owned(),
            mode: VideoMode::Equirect360,
            fps: 2,
            train_steps: 30_000,
            running: false,
            stage: 0,
            done_stages: 0,
            log: Vec::new(),
            rx: None,
            pending_train: None,
            error: None,
        }
    }
}

impl GhiPipelinePanel {
    fn poll_events(&mut self) {
        let Some(rx) = &self.rx else { return };
        let mut disconnect = false;
        loop {
            match rx.try_recv() {
                Ok(PipeEvent::Log(line)) => {
                    self.log.push(line);
                    let len = self.log.len();
                    if len > 500 {
                        self.log.drain(0..len - 500);
                    }
                }
                Ok(PipeEvent::Stage(s)) => {
                    self.stage = s;
                    self.done_stages = s;
                }
                Ok(PipeEvent::Finished(dir)) => {
                    self.running = false;
                    self.done_stages = 4;
                    self.pending_train = Some(dir);
                    disconnect = true;
                }
                Ok(PipeEvent::Failed(err)) => {
                    self.running = false;
                    self.error = Some(err);
                    disconnect = true;
                }
                Err(std::sync::mpsc::TryRecvError::Empty) => break,
                Err(std::sync::mpsc::TryRecvError::Disconnected) => {
                    disconnect = true;
                    break;
                }
            }
        }
        if disconnect {
            self.rx = None;
        }
    }

    fn start_pipeline(&mut self, ctx: egui::Context) {
        let video = PathBuf::from(self.video.trim());
        if self.workdir.trim().is_empty() {
            let stem = video
                .file_stem()
                .map_or_else(|| "video".to_owned(), |s| s.to_string_lossy().into_owned());
            let parent = video.parent().unwrap_or_else(|| Path::new("."));
            self.workdir = parent
                .join(format!("{stem}_workdir"))
                .to_string_lossy()
                .into_owned();
        }

        self.running = true;
        self.stage = 0;
        self.done_stages = 0;
        self.error = None;
        self.pending_train = None;
        self.log.clear();

        let (tx, rx) = channel();
        self.rx = Some(rx);

        let workdir = PathBuf::from(self.workdir.trim());
        let ffmpeg = self.ffmpeg.trim().to_owned();
        let colmap = self.colmap.trim().to_owned();
        let mode = self.mode;
        let fps = self.fps.max(1);

        std::thread::spawn(move || {
            let result = run_pipeline(&tx, &ctx, &video, &workdir, &ffmpeg, &colmap, mode, fps);
            match result {
                Ok(()) => {
                    let _ = tx.send(PipeEvent::Finished(workdir));
                }
                Err(e) => {
                    let _ = tx.send(PipeEvent::Failed(format!("{e:#}")));
                }
            }
            ctx.request_repaint();
        });
    }

    fn start_training(&mut self, dir: &Path, process: &UiProcess) {
        let steps = self.train_steps;
        let source = DataSource::Path(dir.to_string_lossy().into_owned());
        self.log.push(format!(
            "Lanzando entrenamiento ({steps} steps) sobre {}",
            dir.display()
        ));
        process.connect_to_process(create_process(source, move |mut cfg| async move {
            cfg.train_config.total_train_iters = steps;
            Some(cfg)
        }));
    }
}

impl AppPane for GhiPipelinePanel {
    fn title(&self) -> egui::WidgetText {
        "Pipeline GHI".into()
    }

    fn ui(&mut self, ui: &mut egui::Ui, process: &UiProcess) {
        self.poll_events();

        // Chain into training on the UI thread once COLMAP is done.
        if let Some(dir) = self.pending_train.take() {
            self.start_training(&dir, process);
        }

        ui.heading("Vídeo → COLMAP → Splat");
        ui.add_space(6.0);

        egui::Grid::new("ghi_pipeline_grid")
            .num_columns(2)
            .spacing([8.0, 6.0])
            .show(ui, |ui| {
                ui.label("Vídeo (MP4):");
                ui.horizontal(|ui| {
                    ui.add(
                        egui::TextEdit::singleline(&mut self.video).desired_width(260.0),
                    );
                    if ui.button("…").clicked() {
                        if let Some(path) = rfd_pick_video() {
                            self.video = path;
                        }
                    }
                });
                ui.end_row();

                ui.label("Carpeta de trabajo:");
                ui.add(
                    egui::TextEdit::singleline(&mut self.workdir)
                        .hint_text("(auto: <vídeo>_workdir)")
                        .desired_width(290.0),
                );
                ui.end_row();

                ui.label("Tipo de cámara:");
                ui.horizontal(|ui| {
                    ui.selectable_value(&mut self.mode, VideoMode::Equirect360, "360 equirect");
                    ui.selectable_value(&mut self.mode, VideoMode::Pinhole, "Pinhole");
                });
                ui.end_row();

                ui.label("FPS de extracción:");
                ui.add(egui::DragValue::new(&mut self.fps).range(1..=10));
                ui.end_row();

                ui.label("Steps de entrenamiento:");
                ui.add(
                    egui::DragValue::new(&mut self.train_steps)
                        .range(1000..=200_000)
                        .speed(500),
                );
                ui.end_row();

                ui.label("ffmpeg:");
                ui.add(egui::TextEdit::singleline(&mut self.ffmpeg).desired_width(290.0));
                ui.end_row();

                ui.label("COLMAP:");
                ui.add(egui::TextEdit::singleline(&mut self.colmap).desired_width(290.0));
                ui.end_row();
            });

        ui.add_space(8.0);

        let video_ok = !self.video.trim().is_empty() && Path::new(self.video.trim()).is_file();
        ui.horizontal(|ui| {
            let btn = egui::Button::new(if self.running {
                "⏳ Procesando…"
            } else {
                "▶ Procesar"
            });
            if ui.add_enabled(!self.running && video_ok, btn).clicked() {
                self.start_pipeline(ui.ctx().clone());
            }
            if !video_ok && !self.video.trim().is_empty() {
                ui.colored_label(egui::Color32::YELLOW, "el vídeo no existe");
            }
        });

        ui.add_space(8.0);

        for (i, name) in STAGES.iter().enumerate() {
            let (icon, color) = if self.done_stages > i {
                ("✔", egui::Color32::LIGHT_GREEN)
            } else if self.running && self.stage == i {
                ("⏳", egui::Color32::YELLOW)
            } else {
                ("•", egui::Color32::GRAY)
            };
            ui.colored_label(color, format!("{icon} {name}"));
        }

        if let Some(err) = &self.error {
            ui.add_space(6.0);
            ui.colored_label(egui::Color32::LIGHT_RED, format!("ERROR: {err}"));
        }

        ui.add_space(6.0);
        ui.separator();
        egui::ScrollArea::vertical()
            .stick_to_bottom(true)
            .auto_shrink([false, false])
            .show(ui, |ui| {
                for line in &self.log {
                    ui.add(
                        egui::Label::new(egui::RichText::new(line).monospace().size(10.5))
                            .wrap_mode(egui::TextWrapMode::Truncate),
                    );
                }
            });
    }
}

/// Native file picker for the video, separate from rrfd's async API to keep
/// the panel logic synchronous and simple.
fn rfd_pick_video() -> Option<String> {
    // rrfd's pick_file is async and returns a reader; for a plain path the
    // std dialog via PowerShell would be clunky. Reuse rrfd's underlying rfd.
    rfd::FileDialog::new()
        .add_filter("Vídeo", &["mp4", "mov", "insv", "avi", "mkv"])
        .pick_file()
        .map(|p| p.to_string_lossy().into_owned())
}

fn run_pipeline(
    tx: &Sender<PipeEvent>,
    ctx: &egui::Context,
    video: &Path,
    workdir: &Path,
    ffmpeg: &str,
    colmap: &str,
    mode: VideoMode,
    fps: u32,
) -> anyhow::Result<()> {
    let images = workdir.join("images");
    let sparse = workdir.join("sparse");
    let database = workdir.join("database.db");
    std::fs::create_dir_all(&images)?;
    std::fs::create_dir_all(&sparse)?;

    // Stage 0: ffmpeg.
    let _ = tx.send(PipeEvent::Stage(0));
    match mode {
        VideoMode::Equirect360 => {
            // Cube-unwrap into 4 pinhole faces (front/right/back/left),
            // h_fov 100 deg, 1440x1440 — the protocol validated on H03.
            let filter = format!(
                "[0:v]fps={fps},split=4[v1][v2][v3][v4];\
                 [v1]v360=input=e:output=flat:h_fov=100:v_fov=100:yaw=0:pitch=0:w=1440:h=1440[front];\
                 [v2]v360=input=e:output=flat:h_fov=100:v_fov=100:yaw=90:pitch=0:w=1440:h=1440[right];\
                 [v3]v360=input=e:output=flat:h_fov=100:v_fov=100:yaw=180:pitch=0:w=1440:h=1440[back];\
                 [v4]v360=input=e:output=flat:h_fov=100:v_fov=100:yaw=-90:pitch=0:w=1440:h=1440[left]"
            );
            let mut cmd = Command::new(ffmpeg);
            cmd.arg("-y")
                .args(["-hwaccel", "auto"])
                .arg("-i")
                .arg(video)
                .args(["-filter_complex", &filter]);
            for face in ["front", "right", "back", "left"] {
                cmd.args(["-map", &format!("[{face}]")])
                    .args(["-q:v", "3"])
                    .arg(images.join(format!("{face}_%04d.jpg")));
            }
            run_command(tx, ctx, "ffmpeg", cmd)?;
        }
        VideoMode::Pinhole => {
            let mut cmd = Command::new(ffmpeg);
            cmd.arg("-y")
                .args(["-hwaccel", "auto"])
                .arg("-i")
                .arg(video)
                .args(["-vf", &format!("fps={fps}")])
                .args(["-q:v", "3"])
                .arg(images.join("frame_%04d.jpg"));
            run_command(tx, ctx, "ffmpeg", cmd)?;
        }
    }

    // Stage 1: COLMAP feature extraction.
    let _ = tx.send(PipeEvent::Stage(1));
    let mut cmd = Command::new(colmap);
    cmd.arg("feature_extractor")
        .arg("--database_path")
        .arg(&database)
        .arg("--image_path")
        .arg(&images)
        .args(["--ImageReader.single_camera_per_folder", "1"])
        .args(["--FeatureExtraction.use_gpu", "1"]);
    if matches!(mode, VideoMode::Equirect360) {
        // Faces are ideal pinholes: f = 720/tan(50 deg) = 604, c = 720.
        cmd.args(["--ImageReader.camera_model", "PINHOLE"])
            .args(["--ImageReader.single_camera", "1"])
            .args(["--ImageReader.camera_params", "604,604,720,720"]);
    } else {
        cmd.args(["--ImageReader.camera_model", "SIMPLE_RADIAL"])
            .args(["--ImageReader.single_camera", "1"]);
    }
    run_command(tx, ctx, "colmap features", cmd)?;

    // Stage 2: sequential matching with loop detection.
    let _ = tx.send(PipeEvent::Stage(2));
    let mut cmd = Command::new(colmap);
    cmd.arg("sequential_matcher")
        .arg("--database_path")
        .arg(&database)
        .args(["--SequentialMatching.overlap", "20"])
        .args(["--SequentialMatching.loop_detection", "1"])
        .args(["--FeatureMatching.use_gpu", "1"]);
    run_command(tx, ctx, "colmap matching", cmd)?;

    // Stage 3: mapper.
    let _ = tx.send(PipeEvent::Stage(3));
    let mut cmd = Command::new(colmap);
    cmd.arg("mapper")
        .arg("--database_path")
        .arg(&database)
        .arg("--image_path")
        .arg(&images)
        .arg("--output_path")
        .arg(&sparse)
        .args(["--Mapper.ba_global_max_num_iterations", "30"]);
    run_command(tx, ctx, "colmap mapper", cmd)?;

    if !sparse.join("0").join("cameras.bin").is_file() {
        anyhow::bail!("COLMAP no generó sparse/0 — revisa el log (¿pocas imágenes registradas?)");
    }

    let _ = tx.send(PipeEvent::Stage(4));
    Ok(())
}

fn run_command(
    tx: &Sender<PipeEvent>,
    ctx: &egui::Context,
    name: &str,
    mut cmd: Command,
) -> anyhow::Result<()> {
    let _ = tx.send(PipeEvent::Log(format!("── {name} ──")));

    cmd.stdout(Stdio::piped()).stderr(Stdio::piped());

    // Don't flash console windows for child processes.
    #[cfg(windows)]
    {
        use std::os::windows::process::CommandExt;
        const CREATE_NO_WINDOW: u32 = 0x0800_0000;
        cmd.creation_flags(CREATE_NO_WINDOW);
    }

    let mut child = cmd
        .spawn()
        .map_err(|e| anyhow::anyhow!("no se pudo lanzar {name}: {e}"))?;

    let stdout = child.stdout.take();
    let stderr = child.stderr.take();

    let spawn_reader = |stream: Option<Box<dyn std::io::Read + Send>>,
                        tx: Sender<PipeEvent>,
                        ctx: egui::Context| {
        std::thread::spawn(move || {
            use std::io::BufRead;
            let Some(stream) = stream else { return };
            let reader = std::io::BufReader::new(stream);
            let mut since_repaint = 0u32;
            for line in reader.lines().map_while(Result::ok) {
                let trimmed = line.trim_end();
                if trimmed.is_empty() {
                    continue;
                }
                let _ = tx.send(PipeEvent::Log(trimmed.to_owned()));
                since_repaint += 1;
                if since_repaint >= 20 {
                    ctx.request_repaint();
                    since_repaint = 0;
                }
            }
        })
    };

    let h1 = spawn_reader(
        stdout.map(|s| Box::new(s) as Box<dyn std::io::Read + Send>),
        tx.clone(),
        ctx.clone(),
    );
    let h2 = spawn_reader(
        stderr.map(|s| Box::new(s) as Box<dyn std::io::Read + Send>),
        tx.clone(),
        ctx.clone(),
    );

    let status = child.wait()?;
    let _ = h1.join();
    let _ = h2.join();
    ctx.request_repaint();

    if !status.success() {
        anyhow::bail!("{name} terminó con código {:?}", status.code());
    }
    Ok(())
}
