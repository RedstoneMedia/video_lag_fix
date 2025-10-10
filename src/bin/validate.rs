use std::fs::File;
use std::io::{BufRead, BufReader};
use std::path::{Path, PathBuf};
use std::str::FromStr;
use clap::Parser;
use fern::Dispatch;
use video_lag_fix::Args;
use video_lag_fix::find::{find_duplicates};


/// Automatically finds and attempts to fix lags in video recordings with frame interpolation.
#[derive(Parser, Debug, Clone)]
#[clap(author, version, about, long_about = None)]
struct Cli {
    /// The input video file path
    #[arg(short)]
    input_path: PathBuf,

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

    /// Werther to enable debug logging
    #[arg(short, long, action)]
    verbose: bool,
}

impl Cli {
    fn as_args(&self) -> Args {
        Args {
            diff_mean_alpha: self.diff_mean_alpha,
            mul_dup_threshold: self.mul_dup_threshold,
            max_dup_threshold: self.max_dup_threshold,
            min_duplicates: self.min_duplicates,
            max_duplicates: self.max_duplicates,
            recent_motion_mean_alpha: self.recent_motion_mean_alpha,
            slow_motion_mean_alpha: self.slow_motion_mean_alpha,
            motion_compensate_threshold: self.motion_compensate_threshold,
            mul_motion_compensate_threshold_retry: self.mul_motion_compensate_threshold_retry,
            motion_compensate_start: self.motion_compensate_start,
            max_motion_mul: self.max_motion_mul,
            diff_hash_resize: self.diff_hash_resize,
        }
    }
}

fn setup_logging(args: &Cli) {
    Dispatch::new()
        .format(move |out, message, record| {
            out.finish(format_args!("[{}] {}", record.level(), message))
        })
        .level(if args.verbose {log::LevelFilter::Debug} else {log::LevelFilter::Info})
        .chain(std::io::stdout())
        .apply()
        .unwrap();
}

fn main() {
    let cli_args = Cli::parse();
    setup_logging(&cli_args);

    if !cli_args.input_path.is_file() {
        eprintln!("Error: Input file does not exist");
        std::process::exit(1);
    }

    let metadata_path = cli_args.input_path.with_extension("lags.csv");
    let metadata_file = BufReader::new(
        File::open(metadata_path).expect("Should open metadata file")
    );
    let lags : Vec<_> = metadata_file.lines()
        .filter_map(|line| {
            if let Ok(line) = line {
                Some(Lag::from_str(&line).expect("Should parse lag line"))
            } else {
                None
            }
        }).collect();

    let args = cli_args.as_args();
    let stats = validate_lags(&cli_args.input_path, &lags, &args);
    println!("{:#?}", stats);
}

#[derive(Debug)]
struct ValidationStats {
    pub precision: f64,
    pub recall: f64,
    pub f1_score: f64,
    pub true_negatives: usize,
    pub true_positives: usize,
    pub false_negatives: usize,
    pub false_positives: usize,
}

fn validate_lags(input_path: impl AsRef<Path>, lags: &[Lag], args: &Args) -> ValidationStats {
    let found_chains = find_duplicates(input_path, None, args);
    let len = lags.last().map(|l| l.start+l.length).unwrap_or(0)
        .max(found_chains.last()
            .map(|c| *c as usize)
            .unwrap_or(0)
        ) + 1;

    let mut target = vec![false; len];
    for lag in lags {
        for i in 0..lag.length {
            target[lag.start + i] = true;
        }
    }
    let mut found = vec![false; len];
    for frame in found_chains {
        found[frame as usize] = true;
    }

    //println!("{:#?}\n{:#?}", &target, &found);

    let mut true_negatives = 0usize;
    let mut true_positives = 0usize;
    let mut false_negatives = 0usize;
    let mut false_positives = 0usize;
    for (y, x) in target.into_iter().zip(found) {
        match (y, x) {
            (false, false) => {
                true_negatives += 1;
            },
            (true, true) => {
                true_positives += 1;
            },
            (true, false) => {
                false_negatives += 1;
            },
            (false, true) => {
                false_positives += 1;
            }
        }
    }

    let precision = (true_positives as f64) / (true_positives as f64 + false_positives as f64);
    let recall = (true_positives as f64) / (true_positives as f64 + false_negatives as f64);
    let f1_score = 2.0 * (precision * recall) / (precision + recall);

    ValidationStats {
        precision,
        recall,
        f1_score,
        true_negatives,
        true_positives,
        false_negatives,
        false_positives,
    }
}


#[derive(Debug)]
struct Lag {
    pub start: usize,
    pub length: usize,
}

impl FromStr for Lag {
    type Err = &'static str;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        let (s, l) = s.split_once(',').ok_or("Expected comma-separated pair")?;
        let start = s.parse::<usize>().map_err(|_| "Could not parse start")?;
        let length = l.parse::<usize>().map_err(|_| "Could not parse length")?;
        Ok(Self { start, length })
    }
}