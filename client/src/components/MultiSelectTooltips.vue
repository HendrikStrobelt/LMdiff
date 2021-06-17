<template>
  <div class="MultiSelectTooltips">

    <transition-group name="fade">
      <div class="tooltip" v-for="tt in tooltipList"
           :key="tt.index"
           @mouseenter="mouseEnter(tt.index)"
           @mouseleave="mouseEnter(-1)"
           :style="{borderColor:(tt.index===showHoverFor)?'#999':null}"
      >
        <div style="display: flex; flex-direction: row;flex-wrap: nowrap;
      justify-content: space-between; padding: 0 5px;">
          <div>index {{ tt.index }}</div>
          <button @click="closeTT(tt.index)">X</button>
        </div>
        <TooltipContent :tokenization="tt.tokenization"
                        :current-token-info="tt.currentTokenInfo"
        ></TooltipContent>
      </div>
    </transition-group>
  </div>
</template>

<script lang="ts">
import {defineComponent} from "vue";
import {PropType} from "@vue/runtime-core";
import {TokenInfo} from "./InteractiveTokens.vue";
import {Tokenization} from "../etc/tokenization";
import TooltipContent from "./TooltipContent.vue";

export interface ToolTipInfo {
  index: number,
  currentTokenInfo: TokenInfo,
  tokenization: Tokenization
}

export default defineComponent({
  name: "MultiSelectTooltips",
  components: {TooltipContent},
  props: {
    tooltipList: {
      type: Array as PropType<ToolTipInfo[]>
    },
    showHoverFor: {
      type: Number,
      default: -1
    }
  },
  emits: ['closeTT', 'hoverChanged'],
  setup(props, ctx) {
    const closeTT = (index) => {
      ctx.emit('closeTT', {index})
    }
    const mouseEnter = (index) => {
      console.log(index, "--- index");
      ctx.emit('hoverChanged', {index, mini: true})
    }
    return {closeTT, mouseEnter}
  }
})
</script>

<style scoped>
.tooltip {
  border-radius: 20px;
  background-color: #eeeeee;
  display: inline-block;
  padding: 5px;
  margin: 2px;
  border: 2px solid #eeeeee;
}

.fade-enter-active,
.fade-leave-active {
  transition: opacity 0.2s ease;
}

.fade-enter-from,
.fade-leave-to {
  opacity: 0;
}
</style>
