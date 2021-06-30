<template>
  <div class="InteractiveTokens">
    <div class="token" v-for="(t, index) in tokens" :ref="addTokenRef"
         :class="{leftSpace:tokenization.leftSpace(t.token),
         newLine: tokenization.newLine(t.token)  }"
         :key="index"
         :style="{borderBottom: '5px solid',
         backgroundColor:(index===showHoverFor)?t.color:(areSelected.includes(index)?'#ccc':null),
         borderTop:'4px solid',
         // borderTop:(index===showHoverFor)?'3px solid':'0.1px solid',
         // marginTop:(index===showHoverFor)?'0px':'3px', //'#2c2d4d'
         borderBottomColor:(index===showHoverFor)?((areSelected.includes(index))?'#ccc':'#333'):t.color,
         borderTopColor: (areSelected.includes(index))?'#ccc':((index===showHoverFor)?'#333':'rgba(0,0,0,0)'),
         }"
         @mouseenter="mouseEnter(index, $event)"
         @mouseleave="mouseLeave"
         @click="mouseClicked(index)"
    >{{ tokenization.cleanup(t.token) }}
    </div>
    <div class="tooltip" v-if="tt.visible"
         :style="{top:tt.y+'px',
         left:tt.rightAligned?null:tt.x+'px',
         right:tt.rightAligned?tt.x+'px':null,
         borderRadius:tt.rightAligned?'20px 0px 20px 20px':'0 20px 20px 20px'
    }"
    >
      <TooltipContent :current-token-info="currentTokenInfo"
                      :tokenization="tokenization"
                      :showMiniTT="showMiniTT"
      ></TooltipContent>
    </div>

  </div>
</template>
//            .attr('class', d => `token ${d.token.startsWith('Ġ') ? 'spaceLeft' : ''} ${d.token.startsWith('Ċ') ? 'newLine' : ''}`)


<script lang="ts">
import {format} from "d3";
import {throttle} from "lodash";
import {
  defineComponent,
  onBeforeUpdate, onUpdated,
  PropType,
  reactive,
  ref,
  watch
} from "vue";
import {available_tokenizations, Tokenization} from "../etc/tokenization";
import TooltipContent from "./TooltipContent.vue";

export interface ModelTokenInfo {
  prob: number,
  rank: number,
  topk?: [string, number][]
}

export interface TokenInfo {
  token: string,
  index?: number,
  value: number,
  color: string,
  m1?: ModelTokenInfo,
  m2?: ModelTokenInfo,
  diff?: { rank?: number, rank_clamped?: number, prob: number, topk?:number }
}


export default defineComponent({
  name: "InteractiveTokens",
  components: {TooltipContent},
  props: {
    tokens: {
      type: Array as PropType<TokenInfo[]>,
      required: true
    },
    tokenization: {
      type: Object as PropType<Tokenization>,
      default: available_tokenizations.gpt
    },
    showHoverFor: {
      type: Number,
      default: -1
    },
    showMiniTT:{
      type:Boolean,
      default:false
    },
    areSelected: {
      type: Array as PropType<number[]>,
      default: []
    }
  },
  emits: ["hoverChanged", "tokenClicked"],
  setup(props, ctx) {
    const tt = reactive({
      rightAligned: false,
      x: -1,
      y: -1,
      visible: false
    })

    let tokenRefs = [] as Element[];
    const addTokenRef = el => {
      // console.log("---  addtok");
      // console.log(el,"--- el");
      tokenRefs.push(el);
    }
    onBeforeUpdate(() => {
      tokenRefs = []
    })

    const currentTokenInfo = ref(null as TokenInfo);

    const updateTT = index => {
      // console.log(index, "--- index");
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

    const mouseClicked = (index) => {
      ctx.emit('tokenClicked', {index})
    }

    const mouseLeave = () => {
      ctx.emit('hoverChanged', {index: -1})
      // tt.visible = false;
    }

    onUpdated(() => {
      tokenRefs.forEach(r => {
        // console.log("tr--- ");
        // console.log(r.getBoundingClientRect(),"--- r.getBoundingClientRect()");
      })
    })


    return {
      mouseEnter,
      mouseLeave,
      mouseClicked,
      tt,
      currentTokenInfo,
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
  box-sizing: border-box;
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
  background-color: #eee;
  border: 2px solid #2c2d4d;
  padding: 5px;
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
