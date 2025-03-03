use std::ops::{Index, IndexMut};
use smallvec::{Array, SmallVec};

/// Action that can be taken from a state
#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct Action(usize);

impl Action {
    pub fn index(&self) -> usize {
        self.0
    }
}

pub trait State /*: Clone + Copy*/ {

    /// initial state
    fn init() -> Self;

    /// number of possible actions for this state
    fn action_count(&self) -> usize;
    
    /// iterate over the actions
    fn action_iter(&self) -> impl Iterator<Item = Action> + '_ {
        (0..self.action_count()).map(|index| Action(index))
    }

    /// state resulting from taking given action
    fn take(&self, action: Action) -> Self;

    /// reward for terminal state, none for non-terminal state
    fn reward(&self) -> Option<f32>;

    /// value for this state given value for state resulting from taking given action
    fn value(&self, taken: Action, value: f32) -> f32;
}

pub struct ActionMap<T: Array + Sized>(SmallVec<T>);



impl<T: Array<Item = T> + Sized> ActionMap<T> {
    pub fn new(items: impl IntoIterator<Item = T>) -> Self {
        let mut v = SmallVec::<T>::new();
        v.extend(items);
        Self(v)
    }

    pub fn is_empty(&self) -> bool {
        self.0.is_empty()
    }

    pub fn len(&self) -> usize {
        self.0.len()
    }

    pub fn iter(&self) -> impl Iterator<Item = &T> {
        self.0.iter()
    }

    pub fn iter_mut(&mut self) -> impl Iterator<Item = &mut T> {
        self.0.iter_mut()
    }

    pub fn last(&self) -> Option<&T> {
        self.0.last()
    }
}

impl<T: Array<Item = T> + Sized> Index<Action> for ActionMap<T> {
    type Output = T;

    fn index(&self, action: Action) -> &Self::Output {
        &self.0[action.index()]
    }
}

impl<T: Array<Item = T> + Sized> IndexMut<Action> for ActionMap<T> {
    fn index_mut(&mut self, action: Action) -> &mut Self::Output {
        &mut self.0[action.index()]
    }
}