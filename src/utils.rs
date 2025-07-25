use std::path::Path;
use std::time::Duration;
use std::fs;

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