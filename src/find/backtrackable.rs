use std::collections::VecDeque;
use std::fmt::Debug;

pub struct Backtrackable<I, S, T>
where
    I: Iterator<Item = T>,
    S: Copy
{
    iter: I,
    max_history: usize,
    items: VecDeque<Option<T>>,
    states: VecDeque<S>,
}

impl<I, S, T> Backtrackable<I, S, T>
where
    I: Iterator<Item = T>,
    S: Copy + Debug
{
    pub fn new(iter: I, initial_state: S, max_history: usize) -> Self {
        let mut states = VecDeque::with_capacity(max_history);
        states.push_back(initial_state);

        Self {
            iter,
            max_history,
            items: VecDeque::with_capacity(max_history),
            states,
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
        let mut state = self.states.pop_front().expect("Initial state should exist");
        let mut replay_index: Option<usize> = None;
        loop {
            match replay_index {
                None => {
                    let Some(item) = self.iter.next() else {break}; // End of iteration
                    let orig_state = state;
                    let mut ctx = BacktrackCtx::new(&mut self.items, &mut self.states, &mut replay_index, self.max_history);
                    body(&mut state, &item, &mut ctx);
                    let new_backtrack = ctx.new_backtrack;

                    self.items.push_back(Some(item));
                    self.states.push_back(orig_state);
                    assert_eq!(self.states.len(), self.items.len());
                    while self.items.len() > self.max_history {
                        self.items.pop_front();
                        self.states.pop_front();
                    }

                    if let Some(n) = new_backtrack {
                        let new_replay_index = self.items.len() - 1 - n;
                        replay_index = Some(new_replay_index);
                        state = self.states[new_replay_index];
                    }
                },
                Some(mut i) => {
                    self.states[i] = state;

                    let item_ref = self.items.get_mut(i).expect("Item should exist in history");
                    let item = std::mem::take(item_ref).expect("History item should be Some"); // Take
                    let mut ctx = BacktrackCtx::new(&mut self.items, &mut self.states, &mut replay_index, self.max_history);
                    body(&mut state, &item, &mut ctx);

                    if ctx.cleared {
                        i = 0;
                    }
                    if let Some(n) = ctx.new_backtrack {
                        assert_eq!(self.states.len(), self.items.len());
                        self.items[i] = Some(item); // Put back
                        let new_replay_index = i - n;
                        replay_index = Some(new_replay_index);
                        state = self.states[new_replay_index];
                        continue;
                    }

                    assert_eq!(self.states.len(), self.items.len());
                    self.items[i] = Some(item); // Put back

                    // Move to next replay item
                    i += 1;
                    if i >= self.items.len() {
                        replay_index = None;
                        println!("End backtrack");
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

#[allow(dead_code)]
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
        self.replay_index.map(|i| i).unwrap_or(self.items.len())
    }

    /// Get the i-th item from the history (0 = least recently added)
    pub fn get(&self, i: usize) -> Option<&T> {
        if i >= self.len() {
            return None;
        }
        self.items.get(i).map(|i| i.as_ref().expect("Item should exist in history"))
    }

    /// Gets the last item in the history (least recently added)
    pub fn first(&self) -> Option<&T> {
        self.get(0)
    }

    /// Gets the last item in the history (most recently added)
    pub fn last(&self) -> Option<&T> {
        self.get(self.len() - 1)
    }

    pub fn iter(&self) -> impl Iterator<Item = &T> {
        self.items.iter().take(self.len()).filter_map(|i| i.as_ref())
    }

    pub fn clear_and_drain(&mut self) -> VecDeque<T> {
        if self.cleared { return VecDeque::with_capacity(0); }
        if self.new_backtrack.is_some() {
            panic!("Not allowed to backtrack and clear at once");
        }

        let mut removed_items = VecDeque::with_capacity(self.len());
        for _ in 0..self.len() {
            let removed_item = self.items.pop_front()
                .expect("items should not be empty")
                .expect("Item should exist in history");
            removed_items.push_back(removed_item);
            self.states.pop_front();
        }
        self.cleared = true;
        removed_items
    }

    pub fn clear(&mut self) {
        if self.cleared { return; }
        if self.new_backtrack.is_some() {
            panic!("Not allowed to backtrack and clear at once");
        }

        for _ in 0..self.len() {
            self.items.pop_front();
            self.states.pop_front();
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

    // TODO: Add more complex tests so we don't end up with phantom bugs (please not again, this is so bad to debug)

    #[test]
    fn run_and_backtrack_reemits_items() {
        let items = vec![
            'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H'
        ];
        let len = items.len();
        let back_iter = Backtrackable::new(items.into_iter(), State { processed: 0 }, 6);
        let mut saw: Vec<char> = Vec::new();

        let res = back_iter.for_each(|state, cur, ctx| {
            println!("#{} {} {:?}", state.processed, cur, ctx.iter().collect::<Vec<_>>());
            state.processed += 1;
            if *cur == 'B' {
                let old: Vec<_>= ctx.clear_and_drain().into_iter().collect();
                assert_eq!(old, vec!['A']);
            }
            if *cur == 'F' && !saw.contains(&'F') {
                let _ = ctx.backtrack(3); // request to reprocess
            }
            if *cur == 'D' && saw.contains(&'F') {
                let old: Vec<_>= ctx.clear_and_drain().into_iter().collect();
                assert_eq!(old, vec!['B', 'C']);
            }
            saw.push(cur.clone()); // snapshot what we saw
        });
        assert_eq!(res.processed, len);
        assert_eq!(saw, vec!['A', 'B', 'C', 'D', 'E', 'F', 'C', 'D', 'E', 'F', 'G', 'H']);
    }
}
