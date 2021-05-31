<template>
  <NavBar :all-models="allModels" v-model:selected-m1="selectedM1"
          v-model:selected-m2="selectedM2"/>


  <div style="width: 100%; padding: 5px;box-sizing:border-box;">
    <h3>Search for interesting snippets</h3>
    <div style="display: inline-block;">
      <label for="ds-select">Select reference dataset: </label>
      <select name="dataset" id="ds-select" v-model="currentDataset"
              style="background-color: #eee">
        <option v-for="ds in datasets" :key="ds" :value="ds">{{ ds }}</option>
      </select>
    </div>
    <div style="display: inline-block; margin-left: 1em;">
      <label for="metric-select">Select metric: </label>
      <select name="dataset" id="metric-select" v-model="currentMetric"
              style="background-color: #eee">
        <option v-for="metric in availableMetrics" :key="metric"
                :value="metric">{{ metric }}
        </option>
      </select>
    </div>
    <div style="display: inline-block; margin-left: 1em;">
      <button :disabled="!currentDataset.length || states.searchRequestSent"
              @click="searchForSamples">
        search
      </button>
    </div>
    <div v-if="states.searchRequestSent"> Searching ....</div>
    <div v-if="sampleTexts.length>0"
         style="margin-top: 10px; display: flex; flex-wrap: nowrap; flex-direction: column;">
      <div style="margin-bottom: 2px;"> Click on one of the samples to analyze
        in depth:
      </div>
      <div class="sampleText"
           v-for="s in sampleTexts" :key="s.text"
           @click="useSample(s.text)"
      > {{ s.text }} <span
          class="measureNumber"> ({{ s.measure }})</span></div>

    </div>

    <h3 id="Inspector">Inspect text snippet
    </h3>
    <!--suppress HtmlFormInputWithoutLabel -->
    <textarea id="test_text"
              style="width:100%; box-sizing:border-box;border:1px solid;height: 100px;"
              v-model="customText"/>
    <button @click="analyzeText"
            :disabled="states.analyzeRequestSent"
            style="margin: 10px 0"
    > analyze text
    </button>

    <div style="padding: 10px 0" v-show="states.analyzeRequestSent"> Request
      sent and processing .....
    </div>


    <div style="overflow-x: auto;"
         v-show="!states.zeroRequests && !states.analyzeRequestSent">
      <h3> Analysis </h3>
      <div style="margin-top: 10px;">
        Probabilities (larger is better): <br/>
        <LineGraph :values="probValues"
                   @hoverChanged="hoverChanged"
                   :show-hover-for="hoverID"
        ></LineGraph>
      </div>
      <div style="margin-top: 10px;">
        Rank (smaller is better): <br/>
        <LineGraph :values="rankValues" :max-value="2000"
                   :y-scale="scalePow().exponent(.3).clamp(true)"
                   @hoverChanged="hoverChanged"
                   :show-hover-for="hoverID"
        ></LineGraph>
      </div>
      <div style="margin-top: 10px;margin-bottom: 200px;">
        Tokens (select a measure and hover over tokens for detail):
        <div style="display: inline-block">
          <button class="diffMode" v-for="dm in availableDiffModes"
                  :key="dm.key"
                  :class="{selected:dm.key===currentDiffMode}"
                  @click="currentDiffMode = dm.key"
          >{{ dm.name }}
          </button>
        </div>

        <InteractiveTokens :tokens="tokenList"
                           @hoverChanged="hoverChanged"
                           :show-hover-for="hoverID"
        ></InteractiveTokens>

      </div>


    </div>

  </div>
</template>

<script lang="ts">
import {defineComponent, reactive, ref, watch} from 'vue'
import {AnalyzedText, AnalyzeTextResponse, API} from "./api";
import LineGraph from "./components/LineGraph.vue";
import NavBar from "./components/NavBar.vue";
import {scalePow} from "d3"
import diffModes from "./etc/diffModes";
import {get, sortBy} from "lodash"
import InteractiveTokens, {
  ModelTokenInfo,
  TokenInfo
} from "./components/InteractiveTokens.vue";


export default defineComponent({
  name: 'App',
  components: {InteractiveTokens, LineGraph, NavBar},
  setup() {

    const states = reactive({
      analyzeRequestSent: false,
      zeroRequests: true,
      searchRequestSent: false
    })

    const api = new API()
    const allModels = ref([])
    const selectedM1 = ref('')
    const selectedM2 = ref('')
    const probValues = ref([] as number[][])
    const rankValues = ref([] as number[][])
    const hoverID = ref(-1)
    let currentResult = null as AnalyzeTextResponse;

    const availableDiffModes: { key: string, name: string }[] =
        sortBy(Object.entries(diffModes)
            .map(([key, v]) => ({key, name: v.name})), ['key'])
    const currentDiffMode = ref('diff')

    const datasets = ref([] as string[])
    const currentDataset = ref('')

    const availableMetrics = ["avg_rank_diff", "max_rank_diff", "avg_clamped_rank_diff", "max_clamped_rank_diff", "avg_prob_diff", "max_prob_diff", "kl", "avg_topk_diff", "max_topk_diff"]
    const currentMetric = ref("avg_clamped_rank_diff")

    const tokenList = ref([] as any[])


    const hoverChanged = (hover) => {
      hoverID.value = hover.index;
    }


    const customText = ref("The dominant sequence transduction models are based on complex recurrent or convolutional neural networks in an encoder-decoder configuration. " +
        "The best performing models also connect the encoder and decoder through an attention mechanism. We propose a new simple network architecture, the Transformer, based solely on attention mechanisms, dispensing with recurrence and convolutions entirely. Experiments on two machine translation tasks show these models to be superior in quality while being more parallelizable and requiring significantly less time to train. Our model achieves 28.4 BLEU on the WMT 2014 English-to-German translation task, improving over the existing best results, including ensembles by over 2 BLEU. On the WMT 2014 English-to-French translation task, our model establishes a new single-model state-of-the-art BLEU score of 41.8 after training for 3.5 days on eight GPUs, a small fraction of the training costs of the best models from the literature. We show that the Transformer generalizes well to other tasks by applying it successfully to English constituency parsing both with large and limited training data.")

    api.all_projects().then(res => {
      allModels.value = res;
      if (res.length > 1) {
        selectedM1.value = res[0].model;
        selectedM2.value = res[1].model;
      }
    })

    api.all_ds().then(res => {
      datasets.value = res;
      currentDataset.value = res[0];
    })

    const analyzeText = () => {
      states.analyzeRequestSent = true;
      states.zeroRequests = false;
      api.analyze(
          selectedM1.value,
          selectedM2.value,
          customText.value
      ).then(resp => {
        states.analyzeRequestSent = false;
        currentResult = resp;
        updateTokenVis();
        const r = resp.result;
        probValues.value = [r.m1.prob, r.m2.prob]
        rankValues.value = [r.m1.rank, r.m2.rank]


        console.log(resp.result, "--- resp.result");

      })
    }

    const updateTokenVis = async () => {
      const r = currentResult.result;
      // console.log(r,"--- r");
      const modelTokenInfo = (index, modelID = 'm1'): ModelTokenInfo => {
        return {
          prob: r[modelID].prob[index],
          rank: r[modelID].rank[index],
          topk: r[modelID].topk[index].map(d => [d,1]),
        }
      }

      tokenList.value = currentResult.result.tokens.map((token, i) => {
        const value = get(r, diffModes[currentDiffMode.value].access)[i] as number;
        return {
          token,
          value,
          color: diffModes[currentDiffMode.value]
              .colorScale(value),
          m1: modelTokenInfo(i, 'm1'),
          m2: modelTokenInfo(i, 'm2'),
          diff: {
            rank: r.diff.rank[i],
            rank_clamped: r.diff.rank[i],
            prob: r.diff.prob[i]
          }
          // properties: get_prob(i)
        } as TokenInfo
      });
    }

    watch(currentDiffMode, () => {
      updateTokenVis()
    })

    const searchForSamples = () => {
      states.searchRequestSent = true;
      const m1 = selectedM1.value;
      const m2 = selectedM2.value;
      const dataset = currentDataset.value;
      const metric = currentMetric.value;
      api.findSamples(m1, m2, dataset, metric).then(res => {
        console.log(res, "--- res");
        console.log(res.result[0].diff, "--- res.result[0].diff");
        states.searchRequestSent = false;
        sampleTexts.value = res.result.map(s => ({
          text: s.text,
          measure: s.metrics[metric]
        }))

      })
    }

    const sampleTexts = ref([] as { text: string, measure: number }[])

    function scroll(element) {
      const ele = document.getElementById(element);
      window.scrollTo(ele.offsetLeft, ele.offsetTop);
    }

    const useSample = (sampleText) => {
      customText.value = sampleText
      analyzeText();
      scroll('Inspector')
    }


    return {
      allModels,
      selectedM1,
      selectedM2,
      customText,
      analyzeText,
      probValues,
      rankValues,
      scalePow,
      hoverID,
      hoverChanged,
      states,
      availableDiffModes,
      currentDiffMode,
      tokenList,
      datasets,
      currentDataset,
      availableMetrics,
      currentMetric,
      searchForSamples,
      sampleTexts,
      useSample
    }

  }
})
</script>

<style scoped>
.measureNumber {
  font-family: 'IBM Plex Mono', monospace;
  font-size: 9pt;
}

h3 {
  background-color: #eeeeee;
  border-top: 1px solid #2c2d4d;
  color: #2c2d4d;
}

.sampleText {
  /*font-size: 9pt;*/
  cursor: pointer;
  border-left: 3px solid #aaa;
  padding: 2px 5px;
  margin-bottom: 2px;
}

.sampleText:hover {
  background-color: #eeeeee;
}

.diffMode {
  padding: 3px 6px;
  margin: 5px 0;
  display: inline-block;
  /*padding: 8px 3px;*/
  border-left: 1px solid;
  border-top: 1px solid;
  border-bottom: 1px solid;
  border-right: none;
  border-radius: 0;
  border-color: gray;
  /*background: #ccc;*/

}

.diffMode:first-child {
  border-radius: 5px 0 0 5px;
}

.diffMode:last-child {
  border-radius: 0 5px 5px 0;
  border: 1px solid;
}

.diffMode.selected {
  background: #ccc;
}

</style>

<style>
#app {
  /*font-family: Avenir, Helvetica, Arial, sans-serif;*/
  /*-webkit-font-smoothing: antialiased;*/
  /*-moz-osx-font-smoothing: grayscale;*/
  /*text-align: center;*/
  /*color: #2c3e50;*/
  /*margin-top: 60px;*/
}

body {
  background-color: rgb(266, 255, 255);
  font-family: 'IBM Plex Sans', sans-serif;
  /*font-weight: 300;*/
  /*font-size: 10pt;*/
  padding: 0;
  margin: 0;

}

select {
  /*font-size: 9pt;*/
  font-weight: 500;
  background-color: transparent;
  padding: 8px 6px;
  -webkit-box-sizing: border-box;
  -moz-box-sizing: border-box;
  box-sizing: border-box;
  border-radius: 4px;
  border: 0;
  outline: 0;
}

button {
  padding: 8px 12px;
  box-shadow: none;
  border: 1px solid gray;
  border-radius: 10px;
}

button:disabled {
  pointer-events: none;

}

button:hover {
  background: #2c2d4d;
  color: #eeeeee;
}

</style>
