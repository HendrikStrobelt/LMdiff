<template>
  <NavBar :all-models="allModels" v-model:selected-m1="selectedM1"
          v-model:selected-m2="selectedM2"
          @about-clicked="states.showAbout=!states.showAbout"/>

  <transition name="fade">
    <div v-if="states.showAbout" style="max-width: 800px;margin: 0 auto; padding: 15px; box-sizing: border-box;
  background-color: #eee;
  border-radius: 0 0 10px 10px; border: 3px solid #2c2d4d; border-top: none;position: relative;">

      <h3 style="border: none;">About</h3>
      <p>LMdiff allows to observe qualitative differences between language
        models of similar type
        and of the same tokenization method. For a given text, the pairwise
        differences in probability and rank of a token under each of the models
        is encoded in gradients of red (model 1) to blue (model 2). Interesting
        snippets can be searched in a pre-recorded corpus using a
        described metric. For more information please read the NeurIPS demo
        paper on arxive and check out the github repo. </p>
      <p> LMdiff is collaboration between Hendrik Strobelt (MIT-IBM), Benjamin
        Hoover (MIT-IBM),
        Arvind Satyanarayan (MIT-IBM), and Sebastian Gehrmann (HarvardNLP).</p>
      <p> NeurIPS'20 demo paper at Arxive: <a href="">...</a><br/>
        Github Repo: <a href="">...</a><br/>
        2-min intro video: <a href="">...</a><br/>
      </p>
      <div
          style="padding: 2px;cursor: pointer;text-align: center;"
          @click="states.showAbout=false">[close]
      </div>

    </div>
  </transition>
  <div v-if="!states.modelsMatch" style="width:100%; padding: 5px; box-sizing: border-box;
  text-align: center;background-color: #bb1e1e; color:white;">
    -- Please choose compatible models --
  </div>
  <div style="width: 100%; padding: 5px;box-sizing:border-box;" v-else>
    <h3 id="Inspector">
      <svg class="question-icon" v-html="states.showEnter?caretUp:caretDown"
           @click="states.showEnter = !states.showEnter"></svg>
      Enter own text snippet
      <span>
      <svg data-tippy-content="Enter your own text" class="question-icon" v-html="questionMark"></svg>
      </span>
    </h3>

    <transition name="fade2">
      <div v-if="states.showEnter">
    <textarea id="test_text"
              style="width:100%; box-sizing:border-box;border:1px solid lightgray;
              height: 50px;font: inherit;"
              v-model="customText"/>
        <button @click="analyzeText"
                :disabled="states.analyzeRequestSent"
                style="margin: 10px 0"
        > analyze text
        </button>

      </div>
    </transition>
    <h3>
      <svg class="question-icon" v-html="states.showSearch?caretUp:caretDown"
           @click="states.showSearch = !states.showSearch"></svg>
      Or search for interesting snippets
      <svg data-tippy-content="Search for interesting text snippets in pre-analyzed datasets
      using a summary diff measure (like, e.g., average rank diff) per snippet." class="question-icon" v-html="questionMark"></svg>
    </h3>
    <transition name="fade2">
      <div v-if="states.showSearch">
        <div v-if="datasets.length>0">
          <div class="subheading">Select dataset and metric</div>
          <div style="display: inline-block;">
            <!--        <label for="ds-select" class="subheading">Select dataset and metric </label>-->
            <select name="dataset" id="ds-select" v-model="currentDataset"
                    style="background-color: #eee">
              <option v-for="ds in datasets" :key="ds" :value="ds">{{
                  ds
                }}
              </option>
            </select>
          </div>
          <div style="display: inline-block; margin-left: 1em;">
            <!--        <label for="metric-select">Select metric: </label>-->
            <select name="dataset" id="metric-select" v-model="currentMetric"
                    style="background-color: #eee">
              <option v-for="metric in availableMetrics" :key="metric.k"
                      :value="metric.k">{{ metric.d }}
              </option>
            </select>
          </div>
          <div style="display: inline-block; margin-left: 1em;">
            <button
                :disabled="!currentDataset.length || states.searchRequestSent"
                @click="searchForSamples">
              search
            </button>
          </div>
        </div>
        <div v-else> Argh... no snippet dataset available for the current
          selection
          of models.
        </div>
        <div v-if="states.searchRequestSent"> Searching ....</div>
        <div v-if="sampleTexts.length>0"
             style="margin-top: 10px; display: flex; flex-wrap: nowrap; flex-direction: column;">
          <div class="subheading"> Search results (click for details)
          </div>
          <div style="overflow-y: scroll; max-height: 150px;">
            <div class="sampleText"
                 v-for="s in sampleTexts" :key="(s,i)=> s.text+i"
                 @click="useSample(s.text)"
            > {{ s.text }} <span
                class="measureNumber"> ({{ s.measure }})</span></div>
          </div>
        </div>
      </div>
    </transition>

    <div
        style="width:100%; margin: 10px 0; padding: 5px; background: #2c2d4d; color: white;"
        v-show="states.analyzeRequestSent"> Request
      sent and processing .....
    </div>
    <div
        v-show="!states.zeroRequests && !states.analyzeRequestSent">
      <h3> Token Analysis </h3>
      <div style="overflow-x: auto;">
        <div style="margin-top: 10px;">
          <div class="subheading">Probabilities (larger is better)</div>
          <LineGraph :values="probValues"
                     @hoverChanged="hoverChanged"
                     :show-hover-for="hoverID"
          ></LineGraph>
        </div>
        <div style="margin-top: 10px;">
          <div class="subheading">Rank (smaller is better)</div>
          <LineGraph :values="rankValues" :max-value="2000"
                     :y-scale="scalePow().exponent(.3).clamp(true)"
                     @hoverChanged="hoverChanged"
                     :show-hover-for="hoverID"
          ></LineGraph>
        </div>
      </div>
      <div style="margin-top: 10px;">
        <div class="subheading">Measure mapped to each token</div>
        <div style="display: inline-block">
          <button class="diffMode" v-for="dm in availableDiffModes"
                  :key="dm.key"
                  :class="{selected:dm.key===currentDiffMode}"
                  @click="currentDiffMode = dm.key"
                  :data-tippy-content="dm.description"
                  data-tippy-trigger="mouseenter"
          >{{ dm.name }}
          </button>
        </div>
      </div>
      <div style="margin-top: 10px;">
        <div class="subheading">Tokens (hover/click for details)</div>
        <InteractiveTokens :tokens="tokenList"
                           :tokenization="tokenization"
                           @hoverChanged="hoverChanged"
                           @tokenClicked="updateTokenSelection"
                           :are-selected="tooltipList.map(d => d.index)"
                           :show-hover-for="hoverID"
                           :showMiniTT="showMiniTT"
        ></InteractiveTokens>

      </div>
      <div style="margin-top: 20px">
        <div class="subheading">Selected Tokens</div>
        <div v-if="tooltipList.length===0"> (Click on tokens to select some.)
        </div>
        <MultiSelectTooltips :tooltip-list="tooltipList"
                             @closeTT="updateTokenSelection"
                             @hoverChanged="hoverChanged"
                             :show-hover-for="hoverID"
                             v-else
        ></MultiSelectTooltips>
      </div>

    </div>

  </div>
</template>

<script lang="ts">
import {
  defineComponent,
  onUpdated,
  reactive,
  ref,
  watch,
  watchEffect
} from 'vue'
import {AnalyzedText, AnalyzeTextResponse, API, ModelDescription} from "./api";
import LineGraph from "./components/LineGraph.vue";
import NavBar from "./components/NavBar.vue";
import {scalePow, ascending} from "d3"
import diffModes from "./etc/diffModes";
import {get, sortBy, findIndex} from "lodash"
import InteractiveTokens, {
  ModelTokenInfo,
  TokenInfo
} from "./components/InteractiveTokens.vue";
import {available_tokenizations} from "./etc/tokenization";
import MultiSelectTooltips, {ToolTipInfo} from "./components/MultiSelectTooltips.vue";
import {availableMetrics} from "./etc/metrics";
import {caretDown, caretUp, questionMark} from "./etc/symbols";
import tippy from 'tippy.js'
import 'tippy.js/dist/tippy.css';
import {onMounted} from "vue"

export default defineComponent({
  name: 'App',
  components: {MultiSelectTooltips, InteractiveTokens, LineGraph, NavBar},
  setup() {

    const states = reactive({
      analyzeRequestSent: false,
      zeroRequests: true,
      searchRequestSent: false,
      modelsMatch: false,
      showAbout: true,
      showSearch: true,
      showEnter: true
    })

    const api = new API()
    const allModels = ref([] as ModelDescription[])
    const selectedM1 = ref('')
    const selectedM2 = ref('')
    const probValues = ref([] as number[][])
    const rankValues = ref([] as number[][])
    const hoverID = ref(-1)
    const tokenization = ref(available_tokenizations.gpt)
    let currentResult = null as AnalyzeTextResponse;

    const availableDiffModes: { key: string, name: string, description:string }[] =
        sortBy(Object.entries(diffModes)
            .map(([key, v]) => ({key, name: v.name, description: v.description})), ['key'])
    const currentDiffMode = ref('rank_diff_clamped')

    const datasets = ref([] as string[])
    const currentDataset = ref('')


    const currentMetric = ref("avg_clamped_rank_diff")

    const tokenList = ref([] as TokenInfo[])

    const showMiniTT = ref(false);
    const hoverChanged = (hover) => {
      hoverID.value = hover.index;
      showMiniTT.value = !!hover.mini;
    }

    const tooltipList = ref([] as ToolTipInfo[])
    const updateTokenSelection = ({index}) => {
      // TODO: if index does exist delete otherwise add..
      const pos = findIndex(tooltipList.value, d => d.index === index)
      if (pos < 0) {
        //not found
        tooltipList.value = [...tooltipList.value, {
          index,
          currentTokenInfo: tokenList.value[index],
          tokenization: tokenization.value
        } as ToolTipInfo]
            .sort((a, b) => ascending(a.index, b.index))
      } else {
        tooltipList.value.splice(pos, 1);
      }

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

    watchEffect(() => {
      const m1 = allModels.value.filter(d => d.model == selectedM1.value)[0];
      const m2 = allModels.value.filter(d => d.model == selectedM2.value)[0];
      states.modelsMatch = !!selectedM1.value &&
          !!selectedM2.value && (m1.type === m2.type);
      if (states.modelsMatch) {
        tokenization.value = available_tokenizations[m1.token];
        states.zeroRequests = true;
      }
      api.all_ds(selectedM1.value, selectedM2.value).then(res => {
        datasets.value = res;
        if (res.length > 0) currentDataset.value = res[0];
        sampleTexts.value = [];
      })
    })

    const analyzeText = () => {
      tooltipList.value = [];
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
          topk: r[modelID].topk[index],
        }
      }

      tokenList.value = currentResult.result.tokens.map((token, index) => {
        const value = get(r, diffModes[currentDiffMode.value].access)[index] as number;
        return {
          token,
          value,
          index,
          color: diffModes[currentDiffMode.value]
              .colorScale(value),
          m1: modelTokenInfo(index, 'm1'),
          m2: modelTokenInfo(index, 'm2'),
          diff: {
            rank: r.diff.rank[index],
            rank_clamped: r.diff.rank_clamp[index],
            prob: r.diff.prob[index]
          }
          // properties: get_prob(index)
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

    onMounted(() => {
      //
    })

    let tippyStarted = false;
    onUpdated(() => {
      if (!tippyStarted) {
        const t = tippy('[data-tippy-content]', {
          trigger: 'mouseenter click',
        });
        console.log(t, "--- t");
        // tippyStarted = true;
      }
    })

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
      useSample,
      tokenization,
      updateTokenSelection,
      tooltipList,
      showMiniTT,
      questionMark,
      caretUp,
      caretDown
    }

  }
})
</script>

<style scoped>
.question-icon {
  width: 1em;
  height: 1em;
  position: relative;
  top: .13em;
  stroke: #aaa;
  stroke-width: 2;
}

.question-icon:hover {
  stroke: #2c2d4d;
  stroke-width: 2.5;
}

.measureNumber {
  font-family: 'IBM Plex Mono', monospace;
  font-size: 9pt;
}

h3 {
  background-color: #eeeeee;
  /*border-top: 1px solid #2c2d4d;*/
  color: #2c2d4d;
}

.sampleText {
  /*font-size: 9pt;*/
  cursor: pointer;
  border-left: 8px solid #aaa;
  border-bottom: 1px dotted #aaa;
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

.subheading {
  font-weight: bold;
  margin-bottom: 3px;
}

.fade-enter-active,
.fade-leave-active {
  transition: opacity 0.5s ease, transform .5s;
}

.fade-enter-from,
.fade-leave-to {
  opacity: 0;
  transform: translateY(-500px);
  z-index: -1;
}

.fade2-enter-from,
.fade2-leave-to {
  opacity: 0;
  transform-origin: top left;
  transform: scaleY(0.00001);
  z-index: -1;
}

.fade2-enter-active,
.fade2-leave-active {
  transform-origin: top left;
  transition: opacity 0.5s ease, transform .5s;
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
