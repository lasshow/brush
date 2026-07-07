//! GHI Pipeline panel: orchestrates video -> frames (ffmpeg) -> COLMAP (CUDA)
//! -> Brush training, end to end from the GUI.
//!
//! v2 (07-jul-2026): mejoras integradas del pipeline LiDAR+video validado en
//! la oficina — filtro de frames movidos (varianza del Laplaciano), init 3DGS
//! desde nube LiDAR (alineacion Sim3 + coloreado robusto via
//! tools/ghi_lidar_tools.py), modelo de camara OPENCV, resolucion de entreno
//! configurable y post-proceso metrico (splat en metros + version
//! CloudCompare + nube LiDAR coloreada). Sin mascaras de personas a proposito:
//! preferimos fantasmas visibles a huecos negros sin observaciones.
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
    PostDone,
    Failed(String),
}

const ALL_STAGES: [&str; 10] = [
    "Extraer frames (ffmpeg)",
    "Filtrar frames movidos",
    "COLMAP: features",
    "COLMAP: matching",
    "COLMAP: mapper",
    "Modelo → TXT",
    "LiDAR: alinear (Sim3)",
    "LiDAR: colorear (robusto)",
    "LiDAR: init 3DGS",
    "Entrenamiento",
];

/// Parametros inmutables que viajan al hilo de trabajo.
struct PipeParams {
    video: PathBuf,
    workdir: PathBuf,
    ffmpeg: String,
    colmap: String,
    python: String,
    tools: String,
    lidar: String,
    mode: VideoMode,
    fps: u32,
    filter_blur: bool,
}

pub struct GhiPipelinePanel {
    video: String,
    workdir: String,
    ffmpeg: String,
    colmap: String,
    python: String,
    tools: String,
    lidar: String,
    mode: VideoMode,
    fps: u32,
    train_steps: u32,
    max_res: u32,
    max_splats: u32,
    filter_blur: bool,
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
            python: "python".to_owned(),
            tools: r"C:\Users\ilasso\dev\brush\tools\ghi_lidar_tools.py".to_owned(),
            lidar: String::new(),
            mode: VideoMode::Equirect360,
            fps: 2,
            train_steps: 30_000,
            max_res: 1920,
            max_splats: 2_000_000,
            filter_blur: true,
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
    fn lidar_enabled(&self) -> bool {
        !self.lidar.trim().is_empty()
    }

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
                    self.done_stages = ALL_STAGES.len() - 1;
                    self.pending_train = Some(dir);
                    disconnect = true;
                }
                Ok(PipeEvent::PostDone) => {
                    self.running = false;
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

    fn make_params(&mut self) -> PipeParams {
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
        PipeParams {
            video,
            workdir: PathBuf::from(self.workdir.trim()),
            ffmpeg: self.ffmpeg.trim().to_owned(),
            colmap: self.colmap.trim().to_owned(),
            python: self.python.trim().to_owned(),
            tools: self.tools.trim().to_owned(),
            lidar: self.lidar.trim().to_owned(),
            mode: self.mode,
            fps: self.fps.max(1),
        filter_blur: self.filter_blur,
        }
    }

    fn start_pipeline(&mut self, ctx: egui::Context) {
        let params = self.make_params();

        self.running = true;
        self.stage = 0;
        self.done_stages = 0;
        self.error = None;
        self.pending_train = None;
        self.log.clear();

        let (tx, rx) = channel();
        self.rx = Some(rx);

        std::thread::spawn(move || {
            let workdir = params.workdir.clone();
            let result = run_pipeline(&tx, &ctx, &params);
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

    /// Post-proceso metrico: co-registra el ultimo export al marco LiDAR y
    /// escribe splat metros + version CloudCompare + nube LiDAR color.
    fn start_post(&mut self, ctx: egui::Context) {
        let params = self.make_params();
        let (out, stem) = if !params.video.as_os_str().is_empty() {
            (
                params
                    .video
                    .parent()
                    .unwrap_or_else(|| Path::new("."))
                    .to_path_buf(),
                params
                    .video
                    .file_stem()
                    .map_or_else(|| "escena".to_owned(), |s| s.to_string_lossy().into_owned()),
            )
        } else {
            (
                params
                    .workdir
                    .parent()
                    .unwrap_or_else(|| Path::new("."))
                    .to_path_buf(),
                params
                    .workdir
                    .file_name()
                    .map_or_else(|| "escena".to_owned(), |s| s.to_string_lossy().into_owned()),
            )
        };

        self.running = true;
        self.error = None;
        let (tx, rx) = channel();
        self.rx = Some(rx);

        std::thread::spawn(move || {
            let mut cmd = Command::new(&params.python);
            cmd.arg("-u")
                .arg(&params.tools)
                .arg("post")
                .arg("--workdir")
                .arg(&params.workdir)
                .arg("--lidar")
                .arg(&params.lidar)
                .arg("--out")
                .arg(&out)
                .arg("--stem")
                .arg(&stem);
            match run_command(&tx, &ctx, "post-proceso metrico", cmd) {
                Ok(()) => {
                    let _ = tx.send(PipeEvent::Log("✔ post-proceso completado".to_owned()));
                    let _ = tx.send(PipeEvent::PostDone);
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
        let max_res = self.max_res;
        let max_splats = self.max_splats;
        let source = DataSource::Path(dir.to_string_lossy().into_owned());
        self.log.push(format!(
            "Lanzando entrenamiento ({steps} steps, res {max_res}px, max {max_splats} splats) sobre {}",
            dir.display()
        ));
        process.connect_to_process(create_process(source, move |mut cfg| async move {
            cfg.train_config.total_train_iters = steps;
            cfg.train_config.max_splats = max_splats;
            cfg.load_config.max_resolution = max_res;
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

        ui.heading("Vídeo (+LiDAR) → COLMAP → Splat");
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
                        if let Some(path) = rfd_pick_file(&[("Vídeo", &["mp4", "mov", "insv", "avi", "mkv"])]) {
                            self.video = path;
                        }
                    }
                });
                ui.end_row();

                ui.label("Nube LiDAR (PLY):");
                ui.horizontal(|ui| {
                    ui.add(
                        egui::TextEdit::singleline(&mut self.lidar)
                            .hint_text("(opcional — init métrico del splat)")
                            .desired_width(260.0),
                    );
                    if ui.button("…").clicked() {
                        if let Some(path) = rfd_pick_file(&[("Nube de puntos", &["ply"])]) {
                            self.lidar = path;
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

                ui.label("Filtrar frames movidos:");
                ui.checkbox(&mut self.filter_blur, "descartar ~15% menos nítido");
                ui.end_row();

                ui.label("Steps de entrenamiento:");
                ui.add(
                    egui::DragValue::new(&mut self.train_steps)
                        .range(1000..=200_000)
                        .speed(500),
                );
                ui.end_row();

                ui.label("Resolución de entreno:");
                ui.add(
                    egui::DragValue::new(&mut self.max_res)
                        .range(720..=2160)
                        .speed(40)
                        .suffix(" px"),
                );
                ui.end_row();

                ui.label("Máx. splats:");
                ui.add(
                    egui::DragValue::new(&mut self.max_splats)
                        .range(250_000..=5_000_000)
                        .speed(50_000),
                );
                ui.end_row();

                ui.label("ffmpeg:");
                ui.add(egui::TextEdit::singleline(&mut self.ffmpeg).desired_width(290.0));
                ui.end_row();

                ui.label("COLMAP:");
                ui.add(egui::TextEdit::singleline(&mut self.colmap).desired_width(290.0));
                ui.end_row();

                ui.label("python:");
                ui.add(egui::TextEdit::singleline(&mut self.python).desired_width(290.0));
                ui.end_row();

                ui.label("Script LiDAR:");
                ui.add(egui::TextEdit::singleline(&mut self.tools).desired_width(290.0));
                ui.end_row();
            });

        ui.add_space(8.0);

        let video_ok = !self.video.trim().is_empty() && Path::new(self.video.trim()).is_file();
        let lidar_ok = !self.lidar_enabled() || Path::new(self.lidar.trim()).is_file();
        let workdir_set = !self.workdir.trim().is_empty();

        ui.horizontal(|ui| {
            let btn = egui::Button::new(if self.running {
                "⏳ Procesando…"
            } else {
                "▶ Procesar"
            });
            if ui
                .add_enabled(!self.running && video_ok && lidar_ok, btn)
                .clicked()
            {
                self.start_pipeline(ui.ctx().clone());
            }

            // Post-proceso metrico: requiere LiDAR + workdir con align_sim3 y
            // un export de entrenamiento (lo localiza el script).
            let post_ready = !self.running && self.lidar_enabled() && workdir_set;
            if ui
                .add_enabled(post_ready, egui::Button::new("📐 Post-proceso métrico"))
                .on_hover_text(
                    "Tras entrenar: co-registra el splat al marco LiDAR (metros) y escribe\n\
                     <stem>_splat_metros.ply, <stem>_splat_metros_CC.ply (CloudCompare)\n\
                     y <stem>_LiDAR_color_metros.ply junto al vídeo",
                )
                .clicked()
            {
                self.start_post(ui.ctx().clone());
            }

            if !video_ok && !self.video.trim().is_empty() {
                ui.colored_label(egui::Color32::YELLOW, "el vídeo no existe");
            }
            if !lidar_ok {
                ui.colored_label(egui::Color32::YELLOW, "la nube LiDAR no existe");
            }
        });

        ui.add_space(8.0);

        let lidar_on = self.lidar_enabled();
        for (i, name) in ALL_STAGES.iter().enumerate() {
            // Ocultar etapas que no aplican a esta configuracion.
            if i == 1 && !self.filter_blur {
                continue;
            }
            if (5..=8).contains(&i) && !lidar_on {
                continue;
            }
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

/// Native file picker, separate from rrfd's async API to keep the panel
/// logic synchronous and simple.
fn rfd_pick_file(filters: &[(&str, &[&str])]) -> Option<String> {
    let mut dlg = rfd::FileDialog::new();
    for (name, exts) in filters {
        dlg = dlg.add_filter(*name, exts);
    }
    dlg.pick_file().map(|p| p.to_string_lossy().into_owned())
}

/// Varianza del Laplaciano sobre la imagen reescalada: score de nitidez.
fn sharpness_score(path: &Path) -> anyhow::Result<f64> {
    let img = image::open(path)?;
    let g = img
        .resize(512, 512, image::imageops::FilterType::Triangle)
        .to_luma8();
    let (w, h) = g.dimensions();
    if w < 3 || h < 3 {
        return Ok(0.0);
    }
    let px = |x: u32, y: u32| f64::from(g.get_pixel(x, y).0[0]);
    let (mut sum, mut sum2, mut n) = (0.0f64, 0.0f64, 0.0f64);
    for y in 1..h - 1 {
        for x in 1..w - 1 {
            let lap = px(x - 1, y) + px(x + 1, y) + px(x, y - 1) + px(x, y + 1)
                - 4.0 * px(x, y);
            sum += lap;
            sum2 += lap * lap;
            n += 1.0;
        }
    }
    Ok(sum2 / n - (sum / n).powi(2))
}

/// Borra el ~15% de frames menos nitidos (tope 25%) para que los movidos no
/// degraden COLMAP ni el entrenamiento. Mismo criterio validado en la oficina.
fn filter_blurry_frames(
    tx: &Sender<PipeEvent>,
    ctx: &egui::Context,
    images: &Path,
) -> anyhow::Result<()> {
    let mut files: Vec<PathBuf> = std::fs::read_dir(images)?
        .filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| {
            p.extension()
                .is_some_and(|e| e.eq_ignore_ascii_case("jpg"))
        })
        .collect();
    files.sort();
    if files.len() < 20 {
        let _ = tx.send(PipeEvent::Log("pocos frames; se omite el filtro".into()));
        return Ok(());
    }

    let mut scores = Vec::with_capacity(files.len());
    for (i, f) in files.iter().enumerate() {
        scores.push(sharpness_score(f).unwrap_or(0.0));
        if i % 25 == 0 {
            let _ = tx.send(PipeEvent::Log(format!(
                "nitidez {i}/{} frames…",
                files.len()
            )));
            ctx.request_repaint();
        }
    }

    let mut sorted = scores.clone();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let thr = sorted[(sorted.len() as f32 * 0.15) as usize];
    let max_remove = (files.len() as f32 * 0.25) as usize;

    let mut removed = 0usize;
    for (f, s) in files.iter().zip(&scores) {
        if *s < thr && removed < max_remove {
            std::fs::remove_file(f)?;
            removed += 1;
        }
    }
    let _ = tx.send(PipeEvent::Log(format!(
        "filtro de nitidez: {removed} frames movidos descartados de {}",
        files.len()
    )));
    Ok(())
}

fn python_tool(
    tx: &Sender<PipeEvent>,
    ctx: &egui::Context,
    params: &PipeParams,
    name: &str,
    args: &[&str],
) -> anyhow::Result<()> {
    let mut cmd = Command::new(&params.python);
    cmd.arg("-u").arg(&params.tools);
    for a in args {
        cmd.arg(a);
    }
    run_command(tx, ctx, name, cmd)
}

fn run_pipeline(
    tx: &Sender<PipeEvent>,
    ctx: &egui::Context,
    params: &PipeParams,
) -> anyhow::Result<()> {
    let video = &params.video;
    let workdir = &params.workdir;
    let ffmpeg = params.ffmpeg.as_str();
    let colmap = params.colmap.as_str();
    let fps = params.fps;

    let images = workdir.join("images");
    let sparse = workdir.join("sparse");
    let database = workdir.join("database.db");
    std::fs::create_dir_all(&images)?;
    std::fs::create_dir_all(&sparse)?;

    // Stage 0: ffmpeg.
    let _ = tx.send(PipeEvent::Stage(0));
    match params.mode {
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

    // Stage 1: descartar frames movidos (validado: mejora COLMAP y el splat).
    if params.filter_blur {
        let _ = tx.send(PipeEvent::Stage(1));
        filter_blurry_frames(tx, ctx, &images)?;
    }

    // Stage 2: COLMAP feature extraction.
    let _ = tx.send(PipeEvent::Stage(2));
    let mut cmd = Command::new(colmap);
    cmd.arg("feature_extractor")
        .arg("--database_path")
        .arg(&database)
        .arg("--image_path")
        .arg(&images)
        .args(["--ImageReader.single_camera_per_folder", "1"])
        .args(["--FeatureExtraction.use_gpu", "1"]);
    if matches!(params.mode, VideoMode::Equirect360) {
        // Faces are ideal pinholes: f = 720/tan(50 deg) = 604, c = 720.
        cmd.args(["--ImageReader.camera_model", "PINHOLE"])
            .args(["--ImageReader.single_camera", "1"])
            .args(["--ImageReader.camera_params", "604,604,720,720"]);
    } else {
        // OPENCV (radial+tangencial): validado con video de movil 4K; Brush
        // consume la distorsion directamente, sin image_undistorter.
        cmd.args(["--ImageReader.camera_model", "OPENCV"])
            .args(["--ImageReader.single_camera", "1"]);
    }
    run_command(tx, ctx, "colmap features", cmd)?;

    // Stage 3: sequential matching with loop detection.
    let _ = tx.send(PipeEvent::Stage(3));
    let mut cmd = Command::new(colmap);
    cmd.arg("sequential_matcher")
        .arg("--database_path")
        .arg(&database)
        .args(["--SequentialMatching.overlap", "20"])
        .args(["--SequentialMatching.loop_detection", "1"])
        .args(["--FeatureMatching.use_gpu", "1"]);
    run_command(tx, ctx, "colmap matching", cmd)?;

    // Stage 4: mapper.
    let _ = tx.send(PipeEvent::Stage(4));
    let mut cmd = Command::new(colmap);
    cmd.arg("mapper")
        .arg("--database_path")
        .arg(&database)
        .arg("--image_path")
        .arg(&images)
        .arg("--output_path")
        .arg(&sparse)
        .args(["--Mapper.ba_use_gpu", "1"])
        .args(["--Mapper.ba_global_max_num_iterations", "30"]);
    run_command(tx, ctx, "colmap mapper", cmd)?;

    if !sparse.join("0").join("cameras.bin").is_file() {
        anyhow::bail!("COLMAP no generó sparse/0 — revisa el log (¿pocas imágenes registradas?)");
    }

    // Etapas LiDAR opcionales: modelo TXT -> align Sim3 -> color -> init.
    if !params.lidar.is_empty() {
        let _ = tx.send(PipeEvent::Stage(5));
        let sparse_txt = workdir.join("sparse_txt");
        std::fs::create_dir_all(&sparse_txt)?;
        let mut cmd = Command::new(colmap);
        cmd.arg("model_converter")
            .arg("--input_path")
            .arg(sparse.join("0"))
            .arg("--output_path")
            .arg(&sparse_txt)
            .args(["--output_type", "TXT"]);
        run_command(tx, ctx, "modelo → TXT", cmd)?;

        let w = workdir.to_string_lossy().into_owned();
        let _ = tx.send(PipeEvent::Stage(6));
        python_tool(tx, ctx, params, "lidar align",
                    &["align", "--workdir", &w, "--lidar", &params.lidar])?;

        let _ = tx.send(PipeEvent::Stage(7));
        python_tool(tx, ctx, params, "lidar color", &["color", "--workdir", &w])?;

        let _ = tx.send(PipeEvent::Stage(8));
        python_tool(tx, ctx, params, "lidar init", &["init", "--workdir", &w])?;
    }

    let _ = tx.send(PipeEvent::Stage(9));
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
