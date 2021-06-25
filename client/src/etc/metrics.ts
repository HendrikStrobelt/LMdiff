export const availableMetrics = [
  {k: "avg_rank_diff", d: "Average Rank Diff", t: 'rank'},
  {k: "max_rank_diff", d: "Maximum Rank Diff", t: 'rank'},
  {k: "avg_clamped_rank_diff", d: "Average Clamped 50 Rank Diff", t: 'rank'},
  {k: "max_clamped_rank_diff", d: "Maximum Clamped 50 Rank Diff", t: 'rank'},
  {k: "avg_prob_diff", d: "Average Probability Diff", t: 'prob'},
  {k: "max_prob_diff", d: "Maximum Probability Diff", t: 'prob'},
  // "kl",
  {k: "avg_topk_diff", d: "Average TopK List Diff", t: 'topk'},
  {k: "max_topk_diff", d: "Maximum TopK List Diff", t: 'topk'}
]
