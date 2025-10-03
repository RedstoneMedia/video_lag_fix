use std::path::Path;
use std::time::Duration;
use std::fs;
use std::process::Command;

pub const TRY_WAIT_DURATION : Duration = Duration::from_millis(300);
pub const TRY_MAX_TRIES: usize = 3;

pub fn try_delete(path: impl AsRef<Path>, max_tries: usize, wait: Duration) -> std::io::Result<()> {
    let mut i = 0;
    loop {
        match fs::remove_file(path.as_ref()) {
            Ok(_) => return Ok(()),
            Err(e) if i == max_tries - 1 => return Err(e),
            Err(_) => std::thread::sleep(wait),
        }
        i += 1;
    }
}

pub fn try_exists(path: impl AsRef<Path>, max_tries: usize, wait: Duration) -> bool {
    let path = path.as_ref();
    for _ in 0..max_tries {
        if path.exists() {return true};
        std::thread::sleep(wait);
    }
    false
}

pub struct VideoParams {
    pub framerate: f64,
    pub width: u32,
    pub height: u32,
    pub duration: Duration,
}

pub fn get_video_params(file_path: impl AsRef<Path>) -> VideoParams {
    let file_path = file_path.as_ref();
    if !file_path.exists() {
        panic!("File at \"{}\" does not exist", file_path.display());
    }

    let output = Command::new("ffprobe")
        .args([
            "-v", "error",
            "-select_streams", "v:0",
            "-show_entries", "stream=width,height,r_frame_rate",
            "-show_entries", "format=duration",
            "-of", "default=noprint_wrappers=1:nokey=0",
            file_path.to_str().unwrap(),
        ])
        .output()
        .expect("Failed to run ffprobe");

    let stdout = str::from_utf8(&output.stdout).expect("Invalid UTF-8 output from ffprobe");

    let mut width = None;
    let mut height = None;
    let mut framerate = None;
    let mut duration = None;

    for line in stdout.lines() {
        if let Some(val) = line.strip_prefix("width=") {
            width = Some(val.parse().expect("Invalid width"));
        } else if let Some(val) = line.strip_prefix("height=") {
            height = Some(val.parse().expect("Invalid height"));
        } else if let Some(val) = line.strip_prefix("r_frame_rate=") {
            let mut parts = val.splitn(2, '/');
            let num: f64 = parts.next().unwrap_or("0").parse().expect("Invalid numerator");
            let den: f64 = parts.next().unwrap_or("1").parse().expect("Invalid denominator");
            framerate = Some(if den != 0.0 { num / den } else { 0.0 });
        } else if let Some(val) = line.strip_prefix("duration=") {
            let secs = val.parse().expect("Invalid duration");
            duration = Some(Duration::from_secs_f64(secs));
        }
    }

    VideoParams {
        width: width.expect("Missing width"),
        height: height.expect("Missing height"),
        framerate: framerate.expect("Missing framerate"),
        duration: duration.expect("Missing duration"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_get_video_params() {
        let params = get_video_params("tests/test.mkv");
        assert_eq!(params.width, 2560);
        assert_eq!(params.height, 1440);
        assert_eq!(params.framerate, 60.0);
    }
    
}