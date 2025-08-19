use std::collections::VecDeque;
use std::fmt::Debug;

pub struct Backtrackable<I, S, T>
where
    I: Iterator<Item = T>,
    S: Copy
{
    iter: I,
    initial_state: S,
    max_history: usize,
    items: VecDeque<Option<T>>,
    states: VecDeque<S>,
}

impl<I, S, T: Debug> Backtrackable<I, S, T>
where
    I: Iterator<Item = T>,
    S: Copy
{
    pub fn new(iter: I, initial_state: S, max_history: usize) -> Self {
        //let max_history = max_history * 2; (Kind of makes sense if you want the full history while backtracking)
        Self {
            iter,
            initial_state,
            max_history,
            items: VecDeque::with_capacity(max_history),
            states: VecDeque::with_capacity(max_history),
        }
    }

    /// Consume the driver and run the closure for each (possibly re-emitted) item.
    ///
    /// Closure gets:
    /// - &mut S for state
    /// - &T for the current item (zero-copy on the hot path)
    /// - &mut ForEachCtx for history ops and backtracking
    pub fn for_each<F>(mut self, mut body: F) -> S
    where
        F: FnMut(&mut S, &T, &mut BacktrackCtx<S, T>),
    {
        let mut state = self.initial_state;
        let mut replay_index: Option<usize> = None;
        loop {
            self.states.push_back(state);
            while self.states.len() > self.max_history {
                self.states.pop_front();
            }

            match replay_index {
                None => {
                    let Some(item) = self.iter.next() else {break}; // End of iteration
                    let mut ctx = BacktrackCtx::new(&mut self.items, &mut self.states, &mut replay_index, self.max_history);
                    body(&mut state, &item, &mut ctx);
                    let new_backtrack = ctx.new_backtrack;

                    self.items.push_back(Some(item));
                    while self.items.len() > self.max_history {
                        self.items.pop_front();
                    }

                    if let Some(n) = new_backtrack {
                        let new_replay_index = self.items.len() - 1 - n;
                        replay_index = Some(new_replay_index);
                        state = self.states[new_replay_index];
                    }
                },
                Some(mut i) => {
                    let item_ref = self.items.get_mut(i).expect("Item should exist in history");
                    let item = std::mem::take(item_ref).expect("History item should be Some"); // Take
                    let mut ctx = BacktrackCtx::new(&mut self.items, &mut self.states, &mut replay_index, self.max_history);
                    body(&mut state, &item, &mut ctx);
                    if ctx.cleared {
                        i = 0;
                    }
                    if let Some(n) = ctx.new_backtrack {
                        let new_replay_index = i - n;
                        replay_index = Some(new_replay_index);
                        state = self.states[new_replay_index];
                        continue;
                    }
                    self.items[i] = Some(item); // Put back
                    i += 1;

                    // Move to next replay item
                    if i >= self.items.len() {
                        replay_index = None;
                    } else {
                        replay_index = Some(i);
                    }
                }
            }
        }

        state
    }
}

pub struct BacktrackCtx<'a, S, T>
where
    S: Copy
{
    items: &'a mut VecDeque<Option<T>>,
    states: &'a mut VecDeque<S>,
    max_history: usize,
    replay_index: &'a mut Option<usize>,
    new_backtrack: Option<usize>,
    cleared: bool
}

impl<'a, S, T> BacktrackCtx<'a, S, T>
where
    S: Copy
{

    fn new(items: &'a mut VecDeque<Option<T>>, states: &'a mut VecDeque<S>, replay_index: &'a mut Option<usize>, max_history: usize) -> Self {
        Self {
            items,
            states,
            replay_index,
            max_history,
            new_backtrack: None,
            cleared: false
        }
    }

    pub fn len(&self) -> usize {
        self.replay_index.map(|i| i).unwrap_or_else(|| self.items.len())
    }

    pub fn get(&self, i: usize) -> Option<&T> {
        self.items.get(i).expect("Item should exist in history").as_ref()
    }

    pub fn iter(&self) -> impl Iterator<Item = &T> {
        self.items.iter().take(self.len()).filter_map(|i| i.as_ref())
    }

    pub fn pop_oldest_and_clear(&mut self) -> Option<T> {
        let oldest = self.items.pop_front().expect("Item should exist in history");
        self.clear();
        oldest
    }

    pub fn clear(&mut self) {
        if self.cleared { return; }
        for _ in 0..self.len() {
            self.items.pop_front();
            self.states.clear();
        }
        self.cleared = true;
    }

    /// Request backtrack by n steps (applied after the closure returns).
    pub fn backtrack(&mut self, n: usize) {
        if n == 0 {
            panic!("Backtracking zero steps is not allowed");
        }
        if n > self.len() || n >= self.max_history {
            panic!("Not enough history to backtrack {} steps", n);
        }
        self.new_backtrack = Some(n);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[derive(Copy, Clone, Debug, PartialEq, Eq)]
    struct State {
        processed: usize
    }

    #[test]
    fn run_and_backtrack_reemits_items() {
        let items = vec![
            'A', 'B', 'C', 'D', 'E', 'F', 'G'
        ];
        let len = items.len();
        let drv = Backtrackable::new(items.into_iter(), State { processed: 0 }, 6);
        let mut saw: Vec<char> = Vec::new();
        //let mut a = false;

        let res = drv.for_each(|state, cur, ctx| {
            state.processed += 1;
            if *cur == 'E' && !saw.contains(&'E') {
                let _ = ctx.backtrack(3); // request to reprocess
            }
            if *cur == 'C' && saw.contains(&'E') {
                ctx.clear();
            }
            saw.push(cur.clone()); // snapshot what we saw
        });
        assert_eq!(res.processed, len);
        assert_eq!(saw, vec!['A', 'B', 'C', 'D', 'E', 'B', 'C', 'D', 'E', 'F', 'G']);
    }
}
