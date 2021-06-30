import {
  ScaleContinuousNumeric,
  scaleLinear,
  scalePow,
  ScaleThreshold,
  scaleThreshold
} from "d3";

export const standard_scale = scalePow<number, string>().exponent(.3).domain([-1, 0, 1])
  //@ts-ignore
  .range(['#d6604d', '#f7f7f7', '#4393c3']);

const val_standard_scale = scaleLinear();

const uni_scale = scalePow<number, string>().exponent(.3).domain([0, 1])
  //@ts-ignore
  .range(['#f7f7f7', '#808080']);

const val_uni_scale = scaleLinear();

const thScale = scaleThreshold<number, string>().domain([10, 100, 1000]).range([
  '#ffffff',
  '#dddddd',
  '#aaa',
  '#777',
])

const val_thScale = thScale.copy().range(['0-10', '10-100', '100-1000', '1000-']);

const rankDiffScale = scaleThreshold<number, string>()
  .domain([-100, -10, -1, 1, 10, 100])
  .range([
    '#5299cc',
    '#8fb3cc',
    '#b8c4cc',
    '#ffffff',
    '#dbc5c5',
    '#d49d94',
    '#d6604d',
  ])
export const rankDiffScaleClamped = rankDiffScale.copy()
  .domain([-30, -10, -1, 1, 10, 30])

const val_rankDiffScale = rankDiffScale.copy().range(['<-100', '-100 > -10', '-10 > -1', '-1 > 1', '1 > 10', '10 > 100', '>100']);
const val_rankDiffScaleClamped = rankDiffScaleClamped.copy().range(['<-50', '-100 > -10', '-10 > -1', '-1 > 1', '1 > 10', '10 > 100', '>50']);

const top10_scale = scalePow<string, string>().exponent(1).domain([0,10])
  .range(['#fff', '#333']);



const diff_Dimensions: {
  [key: string]:
    {
      colorScale: ScaleContinuousNumeric<any, string> | ScaleThreshold<any, string>,
      valueScale: ScaleContinuousNumeric<any, any> | ScaleThreshold<any, any>,
      discrete: boolean,
      name: string,
      reverse?: boolean,
      diff?: string[],
      access: string,
      description: string
    }
} = {
  diff: {
    discrete: false,
    colorScale: standard_scale,
    valueScale: val_standard_scale,
    name: "Probability Diff",
    description: 'Difference of predicted probabilities: prob(M2) - prob(M1)',
    diff: [standard_scale(-.5), standard_scale(0.5)],
    access: "diff.prob"
  },
  prob_m1: {
    discrete: false,
    colorScale: uni_scale,
    valueScale: val_uni_scale,
    name: "Probability M1",
    description: 'Predicted probabilities under model M1',
    access: 'm1.prob'
  },
  prob_m2: {
    discrete: false,
    colorScale: uni_scale,
    valueScale: val_uni_scale,
    name: "Probability M2",
    description: 'Predicted probabilities under model M2',
    access: 'm2.prob'
  },
  rank_m1: {
    discrete: true,
    colorScale: thScale,
    valueScale: val_thScale,
    name: "Rank M1",
    description: 'Predicted rank under model M1',
    access: 'm1.rank'

  },
  rank_m2: {
    discrete: true,
    colorScale: thScale,
    valueScale: val_thScale,
    name: "Rank M2",
    description: 'Predicted rank under model M2',
    access: 'm2.rank'
  },
  rank_diff: {
    discrete: true,
    reverse: true,
    colorScale: rankDiffScale,
    valueScale: val_rankDiffScale,
    name: "Rank Diff",
    description: 'Difference of predicted rank: rank(M2) - rank(M1)',
    access: 'diff.rank'
  },
  rank_diff_clamped: {
    discrete: true,
    reverse: true,
    colorScale: rankDiffScaleClamped,
    valueScale: val_rankDiffScaleClamped,
    name: "Clamped Rank Diff",
    description: 'Difference of predicted rank clamped to 50: min(rank(M2),50) - min(rank(M1),50)',
    access: 'diff.rank_clamp'
  },
  topk_intersect:{
    discrete:true,
    reverse:false,
    colorScale: top10_scale,
    valueScale: uni_scale,
    name: "Top10 Diff",
    description: '10 - (Size of intersection set of top 10 terms)',
    access: 'diff.topk'
  }
}

export default diff_Dimensions;
