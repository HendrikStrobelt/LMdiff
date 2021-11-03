import * as d3 from 'd3';
import {makeUrl, toPayload} from './etc/apiHelpers'
import * as URLHandler from './etc/urlHandler'

const debug = false;
import response from "./assets/test_analyze.json";


/***
 *  ==== API DATATYPES =====
 */

export type AnalyzedText = {
  prob: {
    diff: number[],
    kl: number[],
    prob_m1: number[],
    prob_m2: number[],
    rank_m1: number[],
    rank_m2: number[]
  }
  tokens: string[]
}

export type SuggestionProps = {
  kl: number[],
  prob_diff: number[],
  prob_m1: number[],
  prob_m2: number[],
  rank_diff: number[],
  rank_diff_clamp: number[],
  rank_m1: number[],
  rank_m2: number[],
  rank_m1_clamp: number[],
  rank_m2_clamp: number[],
  sentence: string,
  tokens: string[],
  topk_m1: [string, number][][],
  topk_m2: [string, number][][],
  inverse_order?: boolean
}

export type AnalyzeResponse = {
  request: { m1: string, m2: string, text: string },
  result: AnalyzedText
}

interface SuggestionResponse {
  request: {
    m1: string, m2: string, corpus: string
  },
  result: {
    [key: string]: {
      probs: {
        clamped_m1_larger_elementwise: SuggestionProps[],
        clamped_m2_larger_elementwise: SuggestionProps[],
        max_elementwise: SuggestionProps[],
        clamped_max_elementwise: SuggestionProps[]
      },
      ranks: {
        clamped_m1_larger_elementwise: SuggestionProps[],
        clamped_m2_larger_elementwise: SuggestionProps[],
        max_elementwise: SuggestionProps[],
        clamped_max_elementwise: SuggestionProps[]
      },
      inverse_order: boolean,
      m1: string,
      m2: string

    }
  }
}

interface PerModelInfo {
  prob: number[];
  rank: number [];
  topk: [string, number] [][];
}

export interface Sample {
  diff: {
    prob: number[],
    rank: number[]
  }
  m1: PerModelInfo,
  m2: PerModelInfo
  example_idx: number,
  metrics: {
    avg_clamped_rank_diff: 10.182
    avg_prob_diff: -0.001
    avg_rank_diff: -487.591
    avg_topk_diff: 3.364
    kl: -0.253
    max_clamped_rank_diff: 46
    max_prob_diff: 0.027
    max_rank_diff: 736
    max_topk_diff: 8
    n_tokens: 22
  },
  text: string,
  tokens: string[]
}

export interface FindSampleResponse {
  request: any,
  result: Sample[],
  histogram: {
    values: number[],
    bin_edges: number[]
  }
}

export interface AnalyzeTextResponse {
  request: { m1: string, m2: string, text: string }
  result: {
    tokens: string[],
    m1: PerModelInfo,
    m2: PerModelInfo,
    diff: { rank: number[], prob: number[], rank_clamp: number[], topk:number[] }
  },
}

export interface SpecificAttentionsResponse {
  m1: number[][][]
  m2: number[][][]
}


export interface ModelDescription {
  model: string,
  type: "gpt" | "bert",
  token: "gpt" | "bert",

  [key: string]: string
}

/**
 * ==== API Object =====
 */
export class API {
  private baseURL: string

  /**
   * @param baseURL The URL for the backend
   * @param apiSubRoute The route to append to backend calles
   */
  constructor(baseURL: string = null, apiSubRoute = "/api") {
    const extension = apiSubRoute == null ? "" : apiSubRoute

    // For monolithic SPAs, assume the backend is running at the same base URL of the webpage
    const base = baseURL == null ? URLHandler.basicURL() : baseURL
    this.baseURL = base + extension;
  }

  /**
   * get a list of all available projects
   */
  public all_projects(): Promise<ModelDescription[]> {
    return d3.json(this.baseURL + '/all-models')
  }

  /**
   * get a list of all available preprocessed datasets
   */
  public all_ds(m1: string = null, m2: string = null): Promise<string[]> {
    const url = makeUrl(this.baseURL + "/available-datasets", {m1, m2})
    return d3.json(url)
  }

  public findSamples(m1: string, m2: string, dataset: string, metric: string,
                     order = 'descending', k = 50): Promise<FindSampleResponse> {
    const payload = {
      m1, m2, dataset, metric, order, k, histogram_bins:30
    }

    return d3.json(makeUrl(this.baseURL + '/new-suggestions', payload))
  }

  public getSpecificAttentions(m1: string, m2: string, text: string, token_index_in_text: number): Promise<SpecificAttentionsResponse> {
    const payload = {
      m1, m2, text, token_index_in_text
    }

    return d3.json(this.baseURL + '/specific-attention', toPayload(payload))
  }

  /**
   * analyze a specific text against M1 and M2
   * @param m1 - Model 1 ID
   * @param m2 - Model 2 ID
   * @param text - the text
   */
  public analyze(m1: string, m2: string, text: string): Promise<AnalyzeTextResponse> {
    const payload = {
      m1, m2, text
    }

    if (debug) {
      return new Promise((resole) => {
        resole(response as any)
      })
    } else {
      return d3.json(this.baseURL + '/analyze-text', {
        method: "POST",
        body: JSON.stringify(payload),
        headers: {
          "Content-type": "application/json; charset=UTF-8"
        }
      });
    }

  }

  public static isDemo = !!import.meta.env.VITE_IS_DEMO

  // /**  Example GET request typed with expected response
  //  *
  //  * @param firstname
  //  */
  // getAHi(firstname: string): Promise<string> {
  //     const toSend = {
  //         firstname: firstname
  //     }
  //
  //     const url = makeUrl(this.baseURL + "/get-a-hi", toSend)
  //     return d3.json(url)
  // }
  //
  // /** Example POST request typed with expected response
  //  *
  //  * @param firstname
  //  */
  // postABye(firstname: string): Promise<string> {
  //     const toSend = {
  //         firstname: firstname,
  //     }
  //
  //     const url = makeUrl(this.baseURL + '/post-a-bye');
  //     const payload = toPayload(toSend)
  //     return d3.json(url, payload)
  // }
};
