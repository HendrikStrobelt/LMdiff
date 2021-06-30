<template>
  <div class="TooltipContent">
    <div class="tooltipSub" v-if="!showMiniTT">
      <div class="tt-c" v-show="!!currentTokenInfo?.m1">
        <div> Prob: {{ formatNumbers(currentTokenInfo?.m1?.prob) }}</div>
        <div> Rank: {{ currentTokenInfo?.m1?.rank }}</div>
        <div style="color:#d6604d"
             :style="{fontWeight:(topk[0]===currentTokenInfo.token)?'bold':null}"
             v-for="topk in (currentTokenInfo?.m1?.topk || [])"
        > {{ formatNumbers(topk[1]) }} - {{ tokenization.cleanup(topk[0]) }}
        </div>
        <!--        <p>{{ currentTokenInfo?.m1 }}</p>-->
      </div>
      <div class="tt-c" v-show="!!currentTokenInfo?.m2">
        <div> Prob: {{ formatNumbers(currentTokenInfo?.m2?.prob) }}</div>
        <div> Rank: {{ currentTokenInfo?.m2?.rank }}</div>
        <div style="color:#4393c3"
             :style="{fontWeight:(topk[0]===currentTokenInfo.token)?'bold':null}"
             v-for="topk in (currentTokenInfo?.m2?.topk || [])"
        > {{ formatNumbers(topk[1]) }} - {{ tokenization.cleanup(topk[0]) }}
        </div>
      </div>
      <div class="tt-c" v-show="!!currentTokenInfo?.diff">
        <div>&Delta;Prob: <span
            :style="{color:conditionalColor(currentTokenInfo?.diff?.prob, false)}">
            {{ formatNumbers(currentTokenInfo?.diff?.prob) }}</span></div>
        <div>&Delta;Rank: <span
            :style="{color:conditionalColor(currentTokenInfo?.diff?.rank)}">
          {{ currentTokenInfo?.diff?.rank }}</span></div>
        <div>&Delta;RankCl: <span
            :style="{color:conditionalColor(currentTokenInfo?.diff?.rank_clamped)}">
          {{ currentTokenInfo?.diff?.rank_clamped }}</span></div>
        <div>&Delta;Top10: <span>
          {{ currentTokenInfo?.diff?.topk }}</span></div>

      </div>

    </div>

    <div
        style="text-align:center;">
      <div style="font-weight: bold; padding: 3px 0;">{{
          tokenization.cleanup(currentTokenInfo?.token)
        }}
      </div>
    </div>
  </div>
</template>

<script lang="ts">
import {defineComponent} from "vue";
import {PropType} from "@vue/runtime-core";
import {TokenInfo} from "./InteractiveTokens.vue";
import {format} from "d3";
import {available_tokenizations, Tokenization} from "../etc/tokenization";

export default defineComponent({
  name: "TooltipContent",
  props: {
    currentTokenInfo: {
      type: Object as PropType<TokenInfo>
    },
    tokenization: {
      type: Object as PropType<Tokenization>,
      default: available_tokenizations.gpt
    },
    showMiniTT:{
      type:Boolean,
      default:false
    },

  },
  setup(props, ctx) {
    const conditionalColor = (x, rank = true) => {
      if (rank)
        return x > 0 ? '#d6604d' : (x < 0 ? '#4393c3' : null)
      else
        return x > 0 ? '#4393c3' : (x < 0 ? '#d6604d' : null)
    }
    const formatNumbers = format('.3f');
    return {conditionalColor, formatNumbers}
  }
})
</script>

<style scoped>
.TooltipContent {
  display: flex;
  flex-direction: column;
  flex-wrap: nowrap;
  pointer-events: none;
  font-family: 'IBM Plex Mono', monospace;
  font-size: 9pt;
}

.tooltipSub {
  display: flex;
  flex-direction: row;
  flex-wrap: nowrap;
}


.tt-c {
  white-space: nowrap;
  padding: 3px 5px;
}
</style>
