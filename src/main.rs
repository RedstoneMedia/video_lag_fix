mod patch;
mod find;
mod rife;
mod utils;

use std::path::PathBuf;
use clap::{CommandFactory, FromArgMatches, Parser};
use mimalloc::MiMalloc;
use rife::{DoneDuplicate, Rife};
use utils::{TRY_MAX_TRIES, TRY_WAIT_DURATION};

pub const RIFE_PATH: &str = "rife-ncnn-vulkan/rife-ncnn-vulkan";

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
    #[arg(short = 'm', default_value = "models/rife-v4.26-large")]
    rife_model_path: PathBuf,

    /// Alpha value for exponential moving average used in difference mean calculation
    /// Range: 0.0..1.0 (lower = smoother, but slower to adapt)
    #[arg(long, default_value_t = 0.16, verbatim_doc_comment)]
    diff_mean_alpha: f32,
    /// Multiplier applied to the average frame difference to calculate the duplicate detection threshold
    /// Range: 0.0..1.0 (lower = less sensitive, more deviation from noise floor required)
    #[arg(long, default_value_t = 0.17, verbatim_doc_comment)]
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
    #[arg(long, default_value_t = 8, verbatim_doc_comment)]
    max_duplicates: usize,

    /// Factor by which input frames are downscaled for perceptual hashing
    /// Range: 1.. (lower = higher hash resolution, more sensitive to small differences)
    #[arg(long, default_value_t = 70, verbatim_doc_comment)]
    diff_hash_resize: u32,

    /// Constant quality value for output video encoder (lower is better)
    #[arg(long, default_value_t = 28)]
    render_cq: u8,
    /// Render preset for output video encoder
    #[arg(long, default_value = "p4")]
    render_preset: String,

    /// Only find duplicates, do not patch video
    #[arg(long, action)]
    find_only: bool
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

    let start = std::time::Instant::now();

    let (duplicate_sender, duplicate_receiver) = std::sync::mpsc::channel::<DoneDuplicate>();
    let mut rife= Rife::start(RIFE_PATH, &args.rife_model_path, move |done_duplicate| {
        //println!("Deleting: {}, {}", input0, input1);
        // Sometimes RIFE still holds the files lock for some reason, even after reporting "done".
        utils::try_delete(&done_duplicate.input0, TRY_MAX_TRIES, TRY_WAIT_DURATION).expect("Failed to remove input file");
        utils::try_delete(&done_duplicate.input1, TRY_MAX_TRIES, TRY_WAIT_DURATION).expect("Failed to remove input file");
        // Tell the patcher to insert this
        duplicate_sender.send(done_duplicate).unwrap();
    });

    if !args.find_only {
        let args_copy = args.clone();
        let patch_thread = std::thread::spawn(move || patch::patch_video(&args_copy, duplicate_receiver));
        find::find_duplicates(&args, &mut rife);
        rife.complete();
        patch_thread.join().unwrap();
    } else {
        find::find_duplicates(&args, &mut rife);
        rife.complete();
    }

    println!("Finished in {:.2}s", start.elapsed().as_secs_f32());
}
