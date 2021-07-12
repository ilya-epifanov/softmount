use clap::Clap;

/// TODO:
#[derive(Clap)]
#[clap()]
pub struct Opts {
    /// Show optic flow debug info in a window
    #[clap(short, long)]
    pub debug_output: Option<f32>,
    #[clap(subcommand)]
    pub subcmd: SubCommand,
}

#[derive(Clap)]
pub enum SubCommand {
    DumpOpticFlow(DumpOpticFlow),
    DumpGyro(DumpGyro),
    Match(Match),
}

/// Export rotation detected from optic flow into csv file
#[derive(Clap)]
pub struct DumpOpticFlow {
    /// Input video file
    #[clap(short, long)]
    pub video: String,
    /// Camera calibration file
    #[clap(short, long)]
    pub camera: String,
    /// Output file name
    #[clap(short, long)]
    pub output: Option<String>,
    /// Dump only cached frames, don't recompute
    #[clap(short, long)]
    pub cached_only: bool,
}

/// Export rotation detected by gyro into csv file
#[derive(Clap)]
pub struct DumpGyro {
    /// Input blackbox file
    #[clap(short, long)]
    pub bbox: String,
    /// Output file name
    #[clap(short, long)]
    pub output: Option<String>,
}

/// Export rotation detected from optic flow into Davinci Resolve spline files
#[derive(Clap)]
pub struct Match {
    /// Input video file
    #[clap(short, long)]
    pub video: String,
    /// Camera calibration file
    #[clap(short, long)]
    pub camera: String,
    /// Input blackbox file
    #[clap(short, long)]
    pub bbox: String,
    /// Blackbox segment index
    #[clap(short, long, default_value = "0")]
    pub bbox_ix: usize,
    /// Output file base name, will be suffixed with '.w.spl', '.x.spl', '.y.spl' and '.z.spl'
    #[clap(short, long)]
    pub output: Option<String>,
    /// Camera uptilt, deg
    #[clap(short, long)]
    pub angle: Option<f64>,
    /// Camera angle search range, deg, up + down, defaults to 10deg if angle was specified, 90deg otherwise
    #[clap(short, long)]
    pub angle_range: Option<f64>,
    /// Smoothing time constant
    #[clap(short, long)]
    pub tau: Option<f64>,
    #[clap(short, long)]
    pub frame_center_offset: Option<f64>,
}
