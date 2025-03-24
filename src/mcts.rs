use crate::env::{Action, ActionMap, State};
use itertools::Itertools;
use ordered_float::OrderedFloat;
use rand::random_range;
use smallvec::Array;
use std::cell::OnceCell;
use std::default::Default;
use std::ops::{Index, IndexMut};

#[derive(Debug, Clone, Copy)]
pub struct NodeRef<const N: usize>(usize);

impl<const N: usize> NodeRef<N> {
    fn index(&self) -> usize {
        self.0
    }
}

unsafe impl<const N: usize> Array for NodeRef<N> {
    type Item = NodeRef<N>;

    fn size() -> usize {
        N
    }
}

pub struct F32<const A: usize>(pub f32);

unsafe impl<const N: usize> Array for F32<N> {
    type Item = F32<N>;

    fn size() -> usize {
        N
    }
}

#[derive(Default)]
pub struct Nodes<
    const N: usize,
    const D: usize,
    S: State<D>,
>(Vec<Node<N, D, S>>);

impl<
    const N: usize,
    const D: usize,
    S: State<D>,
> Nodes<N, D, S> {
    pub fn new() -> (NodeRef<N>, Self) {
        let mut nodes = Self(Vec::new());
        let root_ref = nodes.add_node(S::init());
        (root_ref, nodes)
    }

    /// select an action taking to child node/state
    pub fn select_action(&mut self, node_ref: NodeRef<N>) -> (Action, NodeRef<N>) {
        self.open_if_not(node_ref);
        self[node_ref].select_action(3.0)
    }

    pub fn sample_action(
        &mut self,
        node_ref: NodeRef<N>,
        temperature: f32,
    ) -> (Action, NodeRef<N>) {
        self.open_if_not(node_ref);
        self[node_ref].sample_action(temperature)
    }

    fn add_node(&mut self, state: S) -> NodeRef<N> {
        let index = self.0.len();
        self.0.push(Node::new(state));
        NodeRef(index)
    }

    fn is_open(&self, node_ref: NodeRef<N>) -> bool {
        self[node_ref].has_actions()
    }

    fn open_if_not(&mut self, node_ref: NodeRef<N>) {
        if self.is_open(node_ref) {
            return;
        }
        self.init_actions(node_ref);
        self.add_children(node_ref)
    }

    fn init_actions(&self, node_ref: NodeRef<N>) {
        let mut last_node_ref = self.0.len() - 1;
        self[node_ref].init_actions(|| {
            last_node_ref += 1;
            NodeRef(last_node_ref)
        })
    }

    fn add_children(&mut self, node_ref: NodeRef<N>) {
        let state = &self[node_ref].state;
        let child_states = state
            .action_iter()
            .map(|action| state.take(action))
            .collect_vec();
        child_states.into_iter().for_each(|child_state| {
            _ = self.add_node(child_state);
        })
    }
}

impl<
    const N: usize,
    const D: usize,
    S: State<D>,
> Index<NodeRef<N>> for Nodes<N, D, S> {
    type Output = Node<N, D, S>;

    fn index(&self, index: NodeRef<N>) -> &Self::Output {
        &self.0[index.index()]
    }
}

impl<
    const N: usize,
    const D: usize,
    S: State<D>,
> IndexMut<NodeRef<N>> for Nodes<N, D, S> {
    fn index_mut(&mut self, index: NodeRef<N>) -> &mut Self::Output {
        &mut self.0[index.index()]
    }
}

/// result of taking an action
pub struct Stats<const A: usize> {
    /// N(s, a): The number of times action a has been taken from state s
    count: usize,
    /// W(s, a): The total summed value V obtained from every time we’ve taken action a from
    /// state s in the current search tree
    total_value: f32,
    /// P(s, a): The prior probability of selecting action a from state s; this is the output of
    /// the neural net on state s, except for when s is a root node, where we add Dirichlet
    /// noise (I’ll explain this later) to the neural net output to encourage exploration from the
    /// root node
    prior: f32,
}

impl<const A: usize> ActionMap<Stats<A>> {
    pub fn probability_iter(
        &self,
        visit_count: usize,
        temperature: f32,
    ) -> impl Iterator<Item = F32<A>> + '_ {
        let denom = (visit_count as f32).powf(1.0 / temperature);
        self.iter()
            .map(move |stats| F32((stats.count as f32).powf(1.0 / temperature) / denom))
    }

    pub fn quality(&self) -> f32 {
        self.iter().map(|stats| stats.quality()).sum::<f32>() / self.len() as f32
    }

}

impl<const A: usize> Stats<A> {
    /// Q(s, a): The mean value V obtained from taking action a from state s, equal to W(s, a) / N(s, a)
    pub fn quality(&self) -> f32 {
        self.total_value / self.count as f32
    }

    /// PUCT(s, a) = Q(s, a) + c * P(s, a) * sqrt(N(s))/(1 + N(s, a))
    pub fn puct(&self, visit_count: usize, explore_factor: f32) -> f32 {
        self.quality()
            + explore_factor * self.prior * (visit_count as f32).sqrt() / (1.0 + self.count as f32)
    }
}

unsafe impl<const A: usize> Array for Stats<A> {
    type Item = Stats<A>;

    fn size() -> usize {
        A
    }
}

pub struct Node<
    const N: usize,
    const D: usize,
    S: State<D>,
> {
    state: S,
    /// N(s): The number of times state s has been visited
    visit_count: usize,
    reward: OnceCell<Option<f32>>,
    actions: OnceCell<ActionMap<NodeRef<N>>>,
    action_stats: OnceCell<ActionMap<Stats<N>>>,
}

impl<
    const N: usize,
    const D: usize,
    S: State<D>,
> Node<N, D, S> {
    pub fn new(state: S) -> Self {
        Self {
            state,
            visit_count: 0,
            reward: OnceCell::new(),
            actions: OnceCell::new(),
            action_stats: OnceCell::new(),
        }
    }

    /// the number of times this state has been visited
    pub fn visit_count(&self) -> usize {
        self.visit_count
    }

    /// the reward for this state, none if non-terminal state
    pub fn reward(&self) -> &Option<f32> {
        self.reward.get_or_init(|| self.state.reward())
    }

    pub fn has_actions(&self) -> bool {
        self.action_stats.get().is_some()
    }

    /// initialize the actions
    pub fn init_actions(&self, mut new_node_ref: impl FnMut() -> NodeRef<N>) {
        self.actions.get_or_init(|| {
            let prior = 1.0 / self.state.action_count() as f32;
            let (child_refs, stats): (Vec<_>, Vec<_>) = self
                .state
                .action_iter()
                .map(|_action| {
                    (
                        new_node_ref(),
                        Stats {
                            count: 0,
                            total_value: 0.0,
                            prior,
                        },
                    )
                })
                .unzip();
            self.action_stats.get_or_init(|| ActionMap::new(stats));
            ActionMap::new(child_refs)
        });
    }

    /// get the actions
    fn actions(&self) -> &ActionMap<NodeRef<N>> {
        self.actions.get().expect("actions should be initialized")
    }

    /// get the action stats
    fn action_stats(&self) -> &ActionMap<Stats<N>> {
        self.action_stats
            .get()
            .expect("action stats should be created")
    }

    /// select the action to take
    pub fn select_action(&self, explore_factor: f32) -> (Action, NodeRef<N>) {
        let (action, _) = self
            .action_stats()
            .action_value_iter()
            .map(|(action, stats)| (action, stats.puct(self.visit_count(), explore_factor)))
            .max_by_key(|(_, puct)| OrderedFloat(*puct))
            .expect("at least one action");
        (action, self.actions()[action])
    }

    /// pi_s(a) is probability that action `a` is taken from state `s`
    pub fn action_probability(&self, temperature: f32) -> ActionMap<F32<N>> {
        ActionMap::new(self.action_stats().probability_iter(self.visit_count(), temperature))
    }

    /// sample action
    pub fn sample_action(&self, temperature: f32) -> (Action, NodeRef<N>) {
        let r = random_range(0.0f32..1.0);
        let mut p = 0.0;
        let action = self
            .action_probability(temperature)
            .action_value_iter()
            .find_or_last(|(_action, prob)| {
                p += prob.0;
                p >= r
            })
            .expect("at least one action")
            .0;
        (action, self.actions()[action])
    }

    /// mean quality of state
    pub fn quality(&self) -> f32 {
        self.action_stats().quality()
    }
}

pub struct Tree<
    const N: usize,
    const D: usize,
    S: State<D>,
> {
    root_ref: NodeRef<N>,
    nodes: Nodes<N, D, S>,
}

impl<
    const N: usize,
    const D: usize,
    S: State<D>,
> Default for Tree<N, D, S> {
    fn default() -> Self {
        Self::new()
    }
}

impl<
    const N: usize,
    const D: usize,
    S: State<D>,
> Tree<N, D, S> {
    pub fn new() -> Self {
        let (root_ref, nodes) = Nodes::<N, D, S>::new();
        Self { root_ref, nodes }
    }

    pub fn simulate(&mut self, node_ref: NodeRef<N>, count: usize) {
        let mut back = Vec::with_capacity(128);
        for _ in 0..count {
            let mut curr_ref = node_ref;
            while self.nodes[curr_ref].visit_count() > 0 && self.nodes[curr_ref].reward().is_none()
            {
                let (action, new_curr_ref) = self.nodes.select_action(curr_ref);
                back.push((curr_ref, action));
                curr_ref = new_curr_ref;
            }
            let mut value = match self.nodes[curr_ref].reward() {
                Some(reward) => {
                    // If it is a terminal state, adjudicate it, and set the value V(s) to the game
                    // outcome (player one wins = +1, draw = +0.01, player two wins = -1) for a
                    // two-player game with player one taking action a — (draws get assigned a
                    // small positive value); recall that V(s) => -V(s) for the competing agent
                    // (player two’s values are the negative of player one’s values) if your
                    // situation is a zero-sum game
                    *reward
                }
                None => {
                    // If you have not visited it yet, and it’s not a terminal state, call this a
                    // leaf state. Send the leaf state into the neural net, and initialize the
                    // state’s prior probabilities P(s, a) and value V(s) to whatever the neural net
                    // outputs, adding Dirichlet noise to P(s, a) if s is a root state.
                    // first visit of non-terminal
                    0.0
                }
            };
            // Back-propagate the value V(s) of the leaf or terminal state to the state that
            // led to it, being careful with minus signs; increment the number of times each
            // state-action pair N(s, a) was visited along the path from the root node to the
            // leaf or terminal node; adjust N(s), W(s, a), and Q(s, a) for all the states and
            // actions along this path
            while let Some((prev_ref, action)) = back.pop() {
                let prev = &mut self.nodes[prev_ref];
                prev.visit_count += 1;
                value = prev.state.value(action, value);
                let prev_action_stats = &mut prev
                    .action_stats
                    .get_mut()
                    .expect("action results should be expanded")[action];
                prev_action_stats.count += 1;
                prev_action_stats.total_value += value
            }
        }
    }

    /// execute sim_count simulations
    pub fn execute_episode(
        &mut self,
        sim_count: usize,
        temperature: f32
        /* Neural Net */
    ) -> impl Iterator<Item = Example<N, D>> + '_ {
        let mut cur_ref = self.root_ref;
        let mut back = Vec::new();
        loop {
            self.simulate(cur_ref, sim_count);
            let (action, new_cur_ref) = self.nodes.sample_action(cur_ref, temperature);
            back.push((cur_ref, action));
            cur_ref = new_cur_ref;
            if let Some(reward) = self.nodes[cur_ref].reward() {
                // stop at terminal state
                let mut value = *reward;
                return back.into_iter().map(move |(node_ref, action)| {
                    let node = &self.nodes[node_ref];
                    value = node.state.value(action, value);
                    let state = node.state.as_array();
                    let pi = node.action_probability(temperature);
                    Example{state, pi, value}
                })
            }
        }
    }
}

/// training example
pub struct Example<const N: usize, const D: usize, > {
    pub state: [[i32; D]; D],
    pub pi: ActionMap<F32<N>>,
    pub value: f32,
}