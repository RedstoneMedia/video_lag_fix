use std::fmt::Display;
use std::io::Write;
use std::iter;
use std::path::{Path, PathBuf};
use std::sync::mpsc;
use clap::Parser;
use fern::Dispatch;
use ffmpeg_sidecar::command::FfmpegCommand;
use image::RgbImage;
use rand::{Rng, SeedableRng};
use video_lag_fix::patch::{patch_video, Patch, PatchArgs};
use video_lag_fix::{utils, VIDEO_DECODE_ARGS};

/// Automatically inserts lags for validation.
#[derive(Parser, Debug, Clone)]
#[clap(author, version, about, long_about = None)]
struct Cli {
    /// The input video file path
    #[arg(short)]
    input_path: PathBuf,
    /// The output video file path
    #[arg(short)]
    output_path: PathBuf,

    /// The number of lags to insert
    #[arg(short = 'n')]
    n_inserts: usize,
    /// The maximum number of frames for a lag to last
    #[arg(short, default_value_t = 8)]
    max_length: usize,
    /// The random seed
    #[arg(short)]
    seed: Option<u64>,

    /// The method of hardware acceleration for ffmpeg to use
    #[arg(long, default_value = "cuda")]
    render_hwaccel: Option<String>,
    /// Space seperated output args passed to ffmpeg
    #[arg(short, long, default_value = "-c:v av1_nvenc -preset p5 -rc vbr -cq 36 -rc-lookahead 48 -spatial-aq 1 -aq-strength 10 -multipass 2 -pix_fmt yuv420p -map 0:a -c:a libopus -b:a 48k")]
    render_args: String,

    /// Werther to enable debug logging
    #[arg(short, long, action)]
    verbose: bool,
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

    let patch_args = PatchArgs::new(
        cli_args.render_hwaccel,
        cli_args.render_args.split(' ')
    );
    insert_lags(
        cli_args.input_path,
        cli_args.output_path,
        cli_args.n_inserts,
        cli_args.max_length,
        cli_args.seed,
        &patch_args
    );
}


#[derive(Debug)]
struct Lag {
    pub start: usize,
    pub length: usize,
}

impl Display for Lag {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{},{}", self.start, self.length)
    }
}

pub fn insert_lags(
    input: impl AsRef<Path>,
    output: impl AsRef<Path>,
    n_inserts: usize,
    max_length: usize,
    seed: Option<u64>,
    patch_args: &PatchArgs,
) {
    let input = input.as_ref();
    let output = output.as_ref();

    let video_params = utils::get_video_params(input);
    let n_frames = (video_params.framerate * video_params.duration.as_secs_f64()).floor() as usize;

    let mut rng = if let Some(seed) = seed {
        rand::rngs::StdRng::seed_from_u64(seed)
    } else {
        rand::rngs::StdRng::from_os_rng()
    };

    // Select insert locations
    let mut lags = Vec::with_capacity(n_inserts);
    'outer: while lags.len() < n_inserts {
        let new_start = rng.random_range(1..(n_frames - max_length));
        let new_length = rng.random_range(1..=max_length);// Find frames
        let new_end = new_start + new_length;
        for Lag {start, length} in &lags {
            let end = start + length;
            // +1 because we always want one clean frame between inserts
            if *start <= new_end + 1 && new_start <= end + 1 {
                continue 'outer;
            }
        }

        lags.push(Lag {
            start: new_start,
            length: new_length
        });
    }
    lags.sort_unstable_by_key(|Lag {start, ..}| *start);
    // Write lag metadata
    let meta_path = output.with_extension("lags.csv");
    let mut meta_file = std::fs::File::create(meta_path).expect("Should create metadata file");
    for lag in &lags {
        meta_file.write_all(format!("{}\n", lag).as_bytes()).expect("Should write to metadata file");
    }
    // Select frames to duplicate
    let select_dir = Path::new("tmp/selected");
    select_frames(input, lags.iter().map(|Lag {start, ..}| *start - 1), select_dir);
    // Patch
    let (sender, receiver) = mpsc::channel::<Patch>();
    std::thread::spawn(move || {
        for Lag {start, length} in &lags {
            let frame_path = select_dir.join(format!("{:05}.png", start - 1));
            sender.send(Patch::new(
                *start as u32,
                iter::repeat_n(frame_path, *length).collect()
            )).expect("Should send patch");
        }
    });
    patch_video(input, output, patch_args, receiver);
}

fn select_frames(path: impl AsRef<Path>, target_frames: impl IntoIterator<Item=usize>, select_dir: impl AsRef<Path>) {
    let path = path.as_ref();
    let select_dir = select_dir.as_ref();
    std::fs::create_dir_all(select_dir).expect("Should be able to create directory");

    let mut command = FfmpegCommand::new();
    command.input(path.display().to_string());
    command.args(VIDEO_DECODE_ARGS);
    command.print_command();

    let iter = command.spawn().expect("Ffmpeg should spawn")
        .iter().expect("Should be able to get Ffmpeg event iterator");

    let mut target_frames = target_frames.into_iter().peekable();
    for frame in iter.filter_frames() {
        let Some(next_target) = target_frames.peek() else {break};
        if *next_target == frame.frame_num as usize {
            target_frames.next();
            let img: RgbImage = image::ImageBuffer::from_vec(frame.width, frame.height, frame.data).unwrap();
            let img_path = select_dir.join(format!("{:05}.png", frame.frame_num));
            img.save(img_path).expect("Should save image");
        }
    }
}