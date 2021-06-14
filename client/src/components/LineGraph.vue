<template>
  <svg class="LineGraph" :width="svgDims.width" :height="svgDims.height">
    <g ref="xAxisRef"></g>
    <g ref="yAxisRef"></g>
    <rect
        class="hoverBar"
        v-show="(showHoverFor>=0)"
        :x="showHoverFor*stepSize+intents.l"
        :y="intents.t"
        :height="svgDims.height-intents.t-intents.b"
        :width="stepSize"
    ></rect>
    <g class="lines">
      <path class="line" :d="ld.d" :style="{stroke:ld.stroke}"
            v-for="ld in linedata"></path>
    </g>
    <rect ref="hoverTrigger" :x="intents.l" :y="intents.t"
          :width="svgDims.width-intents.l-intents.r"
          :height="svgDims.height-intents.t-intents.b"
          class="hoverTrigger"
          @mousemove="triggerMove"
          @mouseout="hoverOut"
    ></rect>
  </svg>
</template>

<script lang="ts">
import {defineComponent, PropType, ref, watch, reactive, computed} from "vue";
import {
  line,
  curveStep,
  extent,
  scaleLinear,
  select,
  axisBottom,
  axisLeft, ScaleContinuousNumeric, pointer
} from "d3"
import {flatten} from "lodash"


export default defineComponent({
  name: "LineGraph",
  props: {
    values: {
      type: Array as PropType<number[][]>,
      required: true
    },
    maxValue: {
      type: Number,
      default: Number.MAX_VALUE
    },
    yScale: {
      type: Function as PropType<ScaleContinuousNumeric<number, number>>,
      default: null
    },
    showHoverFor: {
      type: Number,
      default:-1
    }

  },
  emits: ['hoverChanged'],
  setup(props, ctx) {
    const xAxisRef = ref(null as SVGGElement);
    const yAxisRef = ref(null as SVGGElement);
    const hoverTrigger = ref(null as SVGRectElement);
    const svgDims = reactive({width: 100, height: 80})
    const stepSize = 5;
    const intents = {
      l: 40,
      b: 25,
      r: 5,
      t: 5
    };
    const colorPool = d => {
      return ['#d6604d', '#4393c3'][d % 2]
    }

    let triggerMove = ref((e: MouseEvent) => {
    })

    const hoverOut = () =>{
      ctx.emit('hoverChanged', -1)
    }

    const linedata = ref([] as { d: string, stroke: string }[])

    // const width = ref(100);
    // const height = 50;

    watch(() => props.values, (newValues) => {
      console.log("watch--- ");
      const dataLength = newValues ? newValues[0].length : 0;
      const plotLength = dataLength * stepSize;
      const [min, max] = extent(flatten(newValues));

      svgDims.width = intents.l + intents.r + plotLength
      const scaleY = props.yScale ? props.yScale : scaleLinear();
      scaleY.domain([Math.min(0, min), Math.min(max, props.maxValue)])
          .range([svgDims.height - intents.b, intents.t]);

      const scaleX = scaleLinear()
          .domain([0, dataLength])
          .range([intents.l, intents.l + plotLength]);

      triggerMove.value = e => {
        const [x, y] = pointer(e);
        const index = Math.floor(scaleX.invert(x));
        ctx.emit("hoverChanged", {index})
      }

      const lineGen = line<number>()
          .x((d, i) => scaleX(i) + 0.5 * stepSize)
          .y(d => scaleY(d))
          .curve(curveStep);

      linedata.value = newValues.map((data, i) => {
        return {
          d: lineGen(data), stroke: colorPool(i)
        }
      })


      select(xAxisRef.value)
          .call(axisBottom(scaleX))
          .attr('transform', `translate(0,${svgDims.height - intents.b})`)

      select(yAxisRef.value)
          .call(axisLeft(scaleY).ticks(Math.floor(svgDims.height/20)))
          .attr('transform', `translate(${intents.l},0)`)


    })

    // const highlightPos = computed(()=> props.showHoverFor*stepSize)

    return {
      svgDims,
      linedata,
      xAxisRef,
      yAxisRef,
      hoverTrigger,
      intents,
      triggerMove,
      stepSize,
      hoverOut
    }
  }
})
</script>

<style scoped>
.line {
  fill: none;
  stroke-width: 2;
  stroke: #333;
  stroke-opacity: .7;
  pointer-events: none;
}

.hoverTrigger {
  opacity: 0;
  cursor: crosshair;
}

.hoverBar{
  opacity: .3;
  color: #eee;
  pointer-events: none;
}

</style>
