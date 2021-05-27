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

    <div style="padding: 10px 0" v-show="states.analyzeRequestSent">  Request sent and processing .....</div>

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
    </div>

  </div>
</template>

<script lang="ts">
import {defineComponent, reactive, ref} from 'vue'
import HelloWorld from './components/HelloWorld.vue'
import C1 from "./components/c1.vue";
import C2 from "./components/c2.vue";
import {API} from "./api";
import NavBar from "./NavBar.vue";
import LineGraph from "./components/LineGraph.vue";
import {range} from "lodash"
import {scalePow} from "d3"

export default defineComponent({
  name: 'App',
  components: {LineGraph, NavBar},
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

    const hoverChanged = (hover) => {
      hoverID.value = hover.index;
    }


    const customText = ref("The dominant sequence transduction models are based on complex recurrent or convolutional neural networks in an encoder-decoder configuration. The best performing models also connect the encoder and decoder through an attention mechanism. We propose a new simple network architecture, the Transformer, based solely on attention mechanisms, dispensing with recurrence and convolutions entirely. Experiments on two machine translation tasks show these models to be superior in quality while being more parallelizable and requiring significantly less time to train. Our model achieves 28.4 BLEU on the WMT 2014 English-to-German translation task, improving over the existing best results, including ensembles by over 2 BLEU. On the WMT 2014 English-to-French translation task, our model establishes a new single-model state-of-the-art BLEU score of 41.8 after training for 3.5 days on eight GPUs, a small fraction of the training costs of the best models from the literature. We show that the Transformer generalizes well to other tasks by applying it successfully to English constituency parsing both with large and limited training data.")

    api.all_projects().then(res => {
      allModels.value = res;
      if (res.length > 1) {
        selectedM1.value = res[0].model;
        selectedM2.value = res[1].model;
      }
    })

    const analyzeText = () => {
      console.log(selectedM1, selectedM2, "--- selectedM1, selectedM2");

      states.analyzeRequestSent = true;
      states.zeroRequests = false;
      api.analyze(
          selectedM1.value,
          selectedM2.value,
          customText.value
      ).then(resp => {
        states.analyzeRequestSent = false;
        const r = resp.result.prob;
        probValues.value = [r.prob_m1, r.prob_m2]
        rankValues.value = [r.rank_m1, r.rank_m2]

      })


      // probValues.value = [1, 2].map(() => range(150).map(() => Math.random()))
      // rankValues.value = [1, 2].map(() => range(150).map(() => Math.random()))
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
      states
    }

  }
})
</script>

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
  border: 1px solid;
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
