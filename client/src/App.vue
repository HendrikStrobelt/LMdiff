<template>
  <NavBar :all-models="allModels" v-model:selected-m1="selectedM1"
          v-model:selected-m2="selectedM2"/>


  <div style="width: 100%; padding: 5px;box-sizing:border-box;">
    <p>or enter own text:
    </p>
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
      <div style="margin-top: 10px;">
        probabilities: <br/>
        <LineGraph :values="probValues"
                   @hoverChanged="hoverChanged"
                   :show-hover-for="hoverID"
        ></LineGraph>
      </div>
      <div style="margin-top: 10px;">
        rank: <br/>
        <LineGraph :values="rankValues" :max-value="2000"
                   :y-scale="scalePow().exponent(.3).clamp(true)"
                   @hoverChanged="hoverChanged"
                   :show-hover-for="hoverID"
        ></LineGraph>
      </div>
      <div>
        <button class="diffMode" v-for="dm in availableDiffModes"
                :key="dm.key"
                :class="{selected:dm.key===currentDiffMode}"
                @click="currentDiffMode = dm.key"
        >{{ dm.name }}
        </button>
      </div>
      <div style="margin-top: 10px;">
        tokens: <br/>
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
import {AnalyzedText, API} from "./api";
import LineGraph from "./components/LineGraph.vue";
import NavBar from "./components/NavBar.vue";
import {scalePow} from "d3"
import diffModes from "./etc/diffModes";
import {sortBy} from "lodash"
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
      zeroRequests: true
    })

    const api = new API()
    const allModels = ref([])
    const selectedM1 = ref('')
    const selectedM2 = ref('')
    const probValues = ref([] as number[][])
    const rankValues = ref([] as number[][])
    const hoverID = ref(-1)
    let currentResult = null as AnalyzedText;

    const availableDiffModes: { key: string, name: string }[] =
        sortBy(Object.entries(diffModes)
            .map(([key, v]) => ({key, name: v.name})), ['key'])
    const currentDiffMode = ref('diff')

    const tokenList = ref([] as any[])


    const hoverChanged = (hover) => {
      hoverID.value = hover.index;
    }


    const customText = ref("The dominant sequence transduction models are based on complex recurrent or convolutional neural networks in an encoder-decoder configuration. ")// +
    //"The best performing models also connect the encoder and decoder through an attention mechanism. We propose a new simple network architecture, the Transformer, based solely on attention mechanisms, dispensing with recurrence and convolutions entirely. Experiments on two machine translation tasks show these models to be superior in quality while being more parallelizable and requiring significantly less time to train. Our model achieves 28.4 BLEU on the WMT 2014 English-to-German translation task, improving over the existing best results, including ensembles by over 2 BLEU. On the WMT 2014 English-to-French translation task, our model establishes a new single-model state-of-the-art BLEU score of 41.8 after training for 3.5 days on eight GPUs, a small fraction of the training costs of the best models from the literature. We show that the Transformer generalizes well to other tasks by applying it successfully to English constituency parsing both with large and limited training data.")

    api.all_projects().then(res => {
      allModels.value = res;
      if (res.length > 1) {
        selectedM1.value = res[0].model;
        selectedM2.value = res[1].model;
      }
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
        currentResult = resp.result;
        updateTokenVis();
        const r = resp.result.prob;
        probValues.value = [r.prob_m1, r.prob_m2]
        rankValues.value = [r.rank_m1, r.rank_m2]


        console.log(resp.result, "--- resp.result");

      })
    }

    const updateTokenVis = async () => {
      const r = currentResult.prob;
      // console.log(r,"--- r");
      const modelTokenInfo = (index, modelID = 'm1'): ModelTokenInfo => {
        return {
          prob: r['prob_' + modelID][index],
          rank: r['rank_' + modelID][index],
          topk: r['topk_' + modelID][index]
        }
      }

      tokenList.value = currentResult.tokens.map((token, i) => ({
        token,
        value: r[currentDiffMode.value][i] as number,
        color: diffModes[currentDiffMode.value]
            .colorScale(r[currentDiffMode.value][i]),
        m1: modelTokenInfo(i,'m1'),
        m2: modelTokenInfo(i,'m2'),
        diff:{
          rank: r['rank_diff'][i],
          rank_clamped: r['rank_diff_clamped'][i],
          prob: r['diff'][i]
        }
        // properties: get_prob(i)
      } as TokenInfo));
    }

    watch(currentDiffMode, () => {
      updateTokenVis()
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
      tokenList
    }

  }
})
</script>

<style scoped>
.diffMode {
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
