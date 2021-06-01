<template>
  <div class="InteractiveTokens">
    <div class="token" v-for="(t, index) in tokens" :ref="addTokenRef"
         :class="{leftSpace:t.token.startsWith(tokenization.leftSpace),
         newLine: t.token.startsWith(tokenization.newLine)  }"
         :key="index"
         :style="{borderBottom: '5px solid',
         borderBottomColor:t.color,
         backgroundColor:(index===showHoverFor)?t.color:null,
         borderTop:'3px solid',
         marginTop:'2px',
         borderTopColor:(index===showHoverFor)?'#2c2d4d':'rgba(0,0,0,0)',
         }"
         @mouseenter="mouseEnter(index, $event)"
         @mouseleave="mouseLeave"
    >{{ token_cleanup(t.token) }}
    </div>
    <div class="tooltip" v-if="tt.visible"
         :style="{top:tt.y+'px',
         left:tt.rightAligned?null:tt.x+'px',
         right:tt.rightAligned?tt.x+'px':null,
         borderRadius:tt.rightAligned?'20px 0px 20px 20px':'0 20px 20px 20px'
    }"
    >
      <div class="tooltipSub">


        <div class="tt-c" v-show="!!currentTokenInfo?.m1">
          <div> Prob: {{ formatNumbers(currentTokenInfo?.m1?.prob) }}</div>
          <div> Rank: {{ currentTokenInfo?.m1?.rank }}</div>
          <div style="color:#d6604d"
               :style="{fontWeight:(topk[0]===currentTokenInfo.token)?'bold':null}"
               v-for="topk in (currentTokenInfo?.m1?.topk || [])"
          > {{ formatNumbers(topk[1]) }} - {{ token_cleanup(topk[0]) }}
          </div>
          <!--        <p>{{ currentTokenInfo?.m1 }}</p>-->
        </div>
        <div class="tt-c" v-show="!!currentTokenInfo?.m2">
          <div> Prob: {{ formatNumbers(currentTokenInfo?.m2?.prob) }}</div>
          <div> Rank: {{ currentTokenInfo?.m2?.rank }}</div>
          <div style="color:#4393c3"
               :style="{fontWeight:(topk[0]===currentTokenInfo.token)?'bold':null}"
               v-for="topk in (currentTokenInfo?.m2?.topk || [])"
          > {{ formatNumbers(topk[1]) }} - {{ token_cleanup(topk[0]) }}
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
        </div>

      </div>
      <div
           style="text-align:center;">
        <div style="font-weight: bold; padding: 3px 0;">{{ token_cleanup(currentTokenInfo?.token) }}</div>
      </div>
    </div>

  </div>
</template>
//            .attr('class', d => `token ${d.token.startsWith('Ġ') ? 'spaceLeft' : ''} ${d.token.startsWith('Ċ') ? 'newLine' : ''}`)


<script lang="ts">
import {format} from "d3";
import {throttle} from "lodash";
import {
  defineComponent,
  onBeforeUpdate,
  PropType,
  reactive,
  ref,
  watch
} from "vue";
import {token_cleanup} from "../etc/util";

export interface ModelTokenInfo {
  prob: number,
  rank: number,
  topk?: [string, number][]
}

export interface TokenInfo {
  token: string,
  value: number,
  color: string,
  m1?: ModelTokenInfo,
  m2?: ModelTokenInfo,
  diff?: { rank?: number, rank_clamped?: number, prob: number }
}

export interface Tokenization {
  type: string,
  leftSpace: string,
  newLine: string
}

export default defineComponent({
  name: "InteractiveTokens",
  props: {
    tokens: {
      type: Array as PropType<TokenInfo[]>,
      required: true
    },
    tokenization: {
      type: Object as PropType<Tokenization>,
      default: {
        type: 'BPE',
        leftSpace: 'Ġ',
        newLine: 'Ċ'
      }
    },
    showHoverFor: {
      type: Number,
      default: -1
    }
  },
  emits: ["hoverChanged"],
  setup(props, ctx) {
    const tt = reactive({
      rightAligned: false,
      x: -1,
      y: -1,
      visible: false
    })

    let tokenRefs = [] as Element[];
    const addTokenRef = el => {
      // console.log(el,"--- el");
      tokenRefs.push(el);
    }
    onBeforeUpdate(() => {
      tokenRefs = []
    })

    const currentTokenInfo = ref(null as TokenInfo);

    const updateTT = index => {
      console.log(index, "--- index");
      if (index < 0) tt.visible = false;
      else {
        const bb = (tokenRefs[index] as Element).getBoundingClientRect();
        if (bb.left < window.innerWidth / 2) {
          tt.x = bb.left;
          tt.rightAligned = false;
        } else {
          tt.x = window.innerWidth - bb.right;
          tt.rightAligned = true;
        }
        tt.y = bb.bottom + 5;
        tt.visible = true;
        currentTokenInfo.value = props.tokens[index];
      }
    }

    const updateTT_throttled = throttle(updateTT, 100, {leading: false})

    watch(props => props.showHoverFor, (hover) => {
      updateTT_throttled(hover);
    })

    // watch(props => props.tokens, (tokens) => {
    //
    // })
    const mouseEnter = (index, event: MouseEvent) => {
      ctx.emit('hoverChanged', {index})
    }

    const mouseLeave = () => {
      ctx.emit('hoverChanged', {index: -1})
      // tt.visible = false;
    }


    const formatNumbers = format('.3f');
    const conditionalColor = (x, rank = true) => {
      if (rank)
        return x > 0 ? '#d6604d' : (x < 0 ? '#4393c3' : null)
      else
        return x > 0 ? '#4393c3' : (x < 0 ? '#d6604d' : null)
    }

    return {
      mouseEnter,
      mouseLeave,
      token_cleanup,
      tt,
      currentTokenInfo,
      formatNumbers,
      conditionalColor,
      addTokenRef
    }
  }
})
</script>

<style scoped>
.token {
  display: inline-block;
  cursor: crosshair;
  transition: 100ms;
}

.leftSpace {
  margin-left: .5em;
}

.tooltip {
  position: fixed;
  /*width: 10px;*/
  /*height: 10px;*/
  /*background-color: red;*/
  /*border-radius: 10px;*/
  transition: 100ms;
  display: flex;
  flex-direction: column;
  flex-wrap: nowrap;
  pointer-events: none;
  font-family: 'IBM Plex Mono', monospace;
  font-size: 9pt;
  background-color: #eee;
  border: 2px solid #2c2d4d;
  padding: 5px;
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

.fade-enter-active,
.fade-leave-active {
  transition: opacity 0.5s ease;
}

.fade-enter-from,
.fade-leave-to {
  opacity: 0;
}
</style>
