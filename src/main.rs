mod patch;
mod find;
mod rife;
mod utils;

use std::path::PathBuf;
use clap::{CommandFactory, FromArgMatches, Parser};
use fern::Dispatch;
use log::{error, info};
use mimalloc::MiMalloc;
use rife::Rife;
use utils::{TRY_MAX_TRIES, TRY_WAIT_DURATION};
use crate::patch::{Patch, PatchArgs};

pub const RIFE_PATH: &str = "rife-ncnn-vulkan";

#[global_allocator]
static GLOBAL: MiMalloc = MiMalloc;


/// Automatically finds and attempts to fix lags in video recordings with frame interpolation.
#[derive(Parser, Debug, Clone)]
#[clap(author, version, about, long_about = None)]
struct Args {
    /// The input video file path
    #[arg(short)]
    input_path: PathBuf,
    /// The output video file path
    #[arg(short)]
    output_path: PathBuf,

    /// The path to the rife-ncnn-vulkan model relative to the rife binary
    #[arg(short = 'm', default_value = "../models/rife-v4.26-large")]
    rife_model_path: PathBuf,

    /// Alpha value for exponential moving average used in difference mean calculation
    /// Range: 0.0..1.0 (lower = smoother, but slower to adapt)
    #[arg(long, default_value_t = 0.16, verbatim_doc_comment)]
    diff_mean_alpha: f32,
    /// Multiplier applied to the average frame difference to calculate the duplicate detection threshold
    /// Range: 0.0..1.0 (lower = less sensitive, more deviation from noise floor required)
    #[arg(long, default_value_t = 0.165, verbatim_doc_comment)]
    mul_dup_threshold: f32,
    /// Maximum allowed absolute difference between frames to be considered duplicates
    /// Range: 0.0..1.0 (1.0 = completely different frame, 0.0 = hash identical)
    #[arg(long, default_value_t = 0.08, verbatim_doc_comment)]
    max_dup_threshold: f32,
    /// Minimum number of consecutive duplicate frames required to trigger interpolation
    /// Range: 1..
    #[arg(long, default_value_t = 2, verbatim_doc_comment)]
    min_duplicates: usize,
    /// Maximum number of consecutive duplicate frames to interpolate
    /// Range: 1..
    #[arg(long, default_value_t = 12, verbatim_doc_comment)]
    max_duplicates: usize,
    /// Alpha value for exponential moving average used in recent motion calculation (Responds fast)
    /// Range: 0.5..1.0 (lower = smoother, but slower to adapt)
    #[arg(long, default_value_t = 0.7, verbatim_doc_comment)]
    recent_motion_mean_alpha: f32,
    /// Alpha value for exponential moving average used in background motion calculation (Responds slow)
    /// Range: 0.0..0.4 (lower = smoother, but slower to adapt)
    #[arg(long, default_value_t = 0.1, verbatim_doc_comment)]
    slow_motion_mean_alpha: f32,
    /// Multiplier applied to recent motion to calculate the required motion threshold for the motion compensation
    /// Range: 0.0..1.0 (lower = less compensation, higher = more compensation)
    #[arg(long, default_value_t = 0.75, verbatim_doc_comment)]
    motion_compensate_threshold: f32,
    /// Multiplier applied to the motion compensation threshold, to require less motion when retrying a failed motion compensation
    /// Range: 0.0..1.0 (lower = less compensation, higher = more compensation)
    #[arg(long, default_value_t = 0.33, verbatim_doc_comment)]
    mul_motion_compensate_threshold_retry: f32,
    /// How many duplicate frames are needed for the motion compensation to be active
    /// Range: 1..
    #[arg(long, default_value_t = 1, verbatim_doc_comment)]
    motion_compensate_start: usize,
    /// Maximum multiple of the background average motion allowed to still interpolate
    /// Allows for more interpolation in low motion areas while thwarting troubling high motion areas to be interpolated too much
    /// Range: 1.0.. (higher = more motion allowed to be interpolated)
    #[arg(long, default_value_t = 7.3, verbatim_doc_comment)]
    max_motion_mul: f32,

    /// Factor by which input frames are downscaled for perceptual hashing
    /// Range: 1.. (lower = higher hash resolution, more sensitive to small differences)
    #[arg(long, default_value_t = 70, verbatim_doc_comment)]
    diff_hash_resize: u32,

    /// Constant quality value for output video encoder (lower is better)
    #[arg(long, default_value_t = 27)]
    render_cq: u8,
    /// Render preset for output video encoder
    #[arg(long, default_value = "p4")]
    render_preset: String,

    /// Only find duplicates, do not patch video
    #[arg(long, action)]
    find_only: bool,
    /// Werther to enable debug logging
    #[arg(short, long, action)]
    verbose: bool,
}

fn setup_logging(args: &Args) {
    let filename = args.input_path.file_name().expect("Expected input file");
    let log_filename = format!("{}.log", filename.to_string_lossy());
    Dispatch::new()
        .format(move |out, message, record| {
            out.finish(format_args!("[{}] {}", record.level(), message))
        })
        .level(if args.verbose {log::LevelFilter::Debug} else {log::LevelFilter::Info})
        .chain(std::io::stdout())
        .chain(fern::log_file(log_filename).expect("Expected to open log file"))
        .apply()
        .unwrap();
}

fn main() {
    let full_template = "\
{before-help}{name} v{version} by {author}
{about-with-newline}
{usage-heading} {usage}

{all-args}{after-help}";
    let mut matches = Args::command().help_template(full_template).get_matches();
    let args = Args::from_arg_matches_mut(&mut matches).unwrap();
    if !args.input_path.exists() {
        eprintln!("Error: Input file does not exist");
        std::process::exit(1);
    }
    setup_logging(&args);
    info!("Processing: {}", args.input_path.display());

    let start = std::time::Instant::now();

    let (patch_sender, patch_receiver) = std::sync::mpsc::channel::<Patch>();
    let mut rife= Rife::start(RIFE_PATH, &args.rife_model_path, move |done_duplicate| {
        // Sometimes RIFE still holds the files lock for some reason, even after reporting "done".
        utils::try_delete(&done_duplicate.input0, TRY_MAX_TRIES, TRY_WAIT_DURATION).unwrap_or_else(|_| error!("Could not remove {}", done_duplicate.input0));
        utils::try_delete(&done_duplicate.input1, TRY_MAX_TRIES, TRY_WAIT_DURATION).unwrap_or_else(|_| error!("Could not remove {}", done_duplicate.input0));
        // Tell the patcher to insert this
        patch_sender.send(done_duplicate.into()).unwrap();
    });

    if !args.find_only {
        let patch_args = PatchArgs {
            render_cq: args.render_cq,
            render_preset: args.render_preset.clone(),
        };
        let input_path = args.input_path.clone();
        let output_path = args.output_path.clone();
        let patch_thread = std::thread::spawn(move ||
            patch::patch_video(input_path, output_path, &patch_args, patch_receiver)
        );
        std::thread::sleep(std::time::Duration::from_millis(500));

        find::find_duplicates(&args, &mut rife);
        rife.complete();
        patch_thread.join().unwrap();
    } else {
        find::find_duplicates(&args, &mut rife);
        rife.complete();
    }

    info!("Finished in {:.2}s", start.elapsed().as_secs_f32());
}
