//! A map that forgets what was used least recently once its entries
//! cost more than a budget: the bound on the renderers' caches of guest
//! memory (decoded programs and textures) and of host GPU objects built
//! from them (MSL libraries, pipelines, samplers).
//!
//! Every entry such a cache holds can be rebuilt from guest memory, so
//! dropping one changes how long the next use takes, never its result.
//!
//! Eviction runs when an insertion takes the total over the budget. It
//! then drops the least recently used entries until the total is at most
//! three quarters of the budget, so that a cache at its bound does not
//! sort its entries on every insertion. The entry just inserted is never
//! dropped, even if it alone costs more than the budget.

use std::collections::HashMap;
use std::hash::Hash;

/// The share of the budget an eviction brings the total down to.
const LOW_WATER_NUMERATOR: usize = 3;
const LOW_WATER_DENOMINATOR: usize = 4;

struct Slot<V> {
    value: V,
    /// [`Lru::clock`] at the last use.
    used: u64,
    cost: usize,
}

/// Entries by key, each with a cost (bytes, or 1 to bound a count); the
/// least recently used are dropped once the costs add up to more than
/// the budget.
pub struct Lru<K, V> {
    entries: HashMap<K, Slot<V>>,
    /// Counts uses: the order entries were last used in.
    clock: u64,
    cost: usize,
    budget: usize,
}

impl<K: Eq + Hash + Clone, V> Lru<K, V> {
    pub fn new(budget: usize) -> Self {
        Lru {
            entries: HashMap::new(),
            clock: 0,
            cost: 0,
            budget,
        }
    }

    fn tick(&mut self) -> u64 {
        self.clock += 1;
        self.clock
    }

    /// The value of `key`, now the most recently used.
    pub fn get(&mut self, key: &K) -> Option<&V> {
        self.get_mut(key).map(|value| &*value)
    }

    /// The value of `key`, now the most recently used.
    pub fn get_mut(&mut self, key: &K) -> Option<&mut V> {
        let now = self.tick();
        let slot = self.entries.get_mut(key)?;
        slot.used = now;
        Some(&mut slot.value)
    }

    /// Holds `value` under `key` (replacing what was there), then drops
    /// least recently used entries if the budget is exceeded; returns
    /// what was dropped (not the replaced value).
    pub fn insert(&mut self, key: K, value: V, cost: usize) -> Vec<(K, V)> {
        let used = self.tick();
        if let Some(old) = self.entries.insert(key.clone(), Slot { value, used, cost }) {
            self.cost -= old.cost;
        }
        self.cost += cost;
        if self.cost <= self.budget {
            return Vec::new();
        }
        let target = self.budget / LOW_WATER_DENOMINATOR * LOW_WATER_NUMERATOR;
        let mut order: Vec<(u64, K)> = self
            .entries
            .iter()
            .filter(|(k, _)| **k != key)
            .map(|(k, slot)| (slot.used, k.clone()))
            .collect();
        order.sort_unstable_by_key(|(used, _)| *used);
        let mut dropped = Vec::new();
        for (_, k) in order {
            if self.cost <= target {
                break;
            }
            let slot = self.entries.remove(&k).expect("listed above");
            self.cost -= slot.cost;
            dropped.push((k, slot.value));
        }
        dropped
    }

    /// Keeps only the entries `keep` accepts.
    pub fn retain(&mut self, mut keep: impl FnMut(&K, &V) -> bool) {
        let mut cost = self.cost;
        self.entries.retain(|k, slot| {
            let kept = keep(k, &slot.value);
            if !kept {
                cost -= slot.cost;
            }
            kept
        });
        self.cost = cost;
    }

    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// The costs of all entries, added.
    pub fn cost(&self) -> usize {
        self.cost
    }
}

#[cfg(test)]
mod tests {
    use super::Lru;

    #[test]
    fn drops_least_recently_used_down_to_three_quarters() {
        let mut lru = Lru::new(8);
        for key in 0..8 {
            assert!(lru.insert(key, key * 10, 1).is_empty());
        }
        // Uses make 0 and 1 the most recent.
        assert_eq!(lru.get(&0), Some(&0));
        assert_eq!(lru.get_mut(&1).copied(), Some(10));
        let mut dropped: Vec<i32> = lru.insert(8, 80, 1).into_iter().map(|(k, _)| k).collect();
        dropped.sort();
        // 9 entries, low water 6: the three least recently used go.
        assert_eq!(dropped, [2, 3, 4]);
        assert_eq!(lru.len(), 6);
        assert_eq!(lru.cost(), 6);
        for key in [0, 1, 5, 6, 7, 8] {
            assert!(lru.get(&key).is_some(), "{key} kept");
        }
    }

    #[test]
    fn costs_replacement_and_oversized_entries() {
        let mut lru = Lru::new(100);
        lru.insert("a", 1, 40);
        lru.insert("b", 2, 40);
        // Replacing an entry counts its new cost only.
        assert!(lru.insert("a", 3, 50).is_empty());
        assert_eq!(lru.cost(), 90);
        // An entry over the whole budget stays; everything else goes.
        let dropped = lru.insert("c", 4, 200);
        assert_eq!(dropped.len(), 2);
        assert_eq!(lru.len(), 1);
        assert_eq!(lru.get(&"c"), Some(&4));
        assert_eq!(lru.cost(), 200);
        lru.retain(|_, _| false);
        assert_eq!((lru.len(), lru.cost()), (0, 0));
    }
}
