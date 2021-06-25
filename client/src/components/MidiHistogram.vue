<template>
  <svg :width="size.width" :height="size.height" class="MidiHistogram"
       ref="myself">
    <g class="xAxis axis" ref="xAxis"
       :transform="`translate(${margin.l}, ${size.height-margin.b})`">
    </g>
    <g class="yAxis axis" ref="yAxis"
       :transform="`translate(${margin.l}, ${margin.t})`"></g>
    <g :transform="`translate(${margin.l}, ${margin.t})`">
      <rect v-for="bar in bars" class="bar"
            :x="bar.x" :y="bar.y" :width="bar.w" :height="bar.h"
            :style="{fill:bar.color?bar.color:null}"
            :data-tippy-content="bar.label"
      ></rect>
    </g>
  </svg>
</template>

<script lang="ts">
import {
  defineComponent,
  onMounted,
  onUpdated,
  ref,
  watch,
  watchEffect
} from "vue";
import {PropType} from "@vue/runtime-core";
import {probDiffColors} from "../etc/colors";
import {axisBottom, axisLeft, extent, scaleLinear, select} from "d3";
import tippy from "tippy.js"

interface BarRender {
  x: number,
  y: number,
  w: number,
  h: number,
  color?: string,
  label: string
}


export default defineComponent({
  name: "MidiHistogram",
  props: {
    values: {
      type: Array as PropType<number[]>,
      required: true
    },
    binEdges: {
      type: Array as PropType<number[]>,
      required: true
    },
    colorScheme: {
      type: Array as PropType<string[]>,
      default: probDiffColors
    },
    height: {
      type: Number,
      default: 150
    },
    width: {
      type: Number,
      default: 200
    }
  },
  setup(props, ctx) {
    const size = {
      width: props.width,
      height: props.height,
    }
    const margin = {
      l: 45,
      r: 5,
      b: 22,
      t: 5
    }

    const bars = ref([] as BarRender[]);
    const xAxis = ref(null as SVGGElement);
    const yAxis = ref(null as SVGGElement);
    const myself = ref(null as SVGViewElement)


    const updateVis = () => {
      const xScale = scaleLinear()
          // extent(props.binEdges)
          .domain([0, props.binEdges.length])
          .range([0, size.width - margin.l - margin.r])
      const yScale = scaleLinear()
          .domain([0, Math.max(...props.values)])
          .range([size.height - margin.b - margin.t, 0])

      const maxV = Math.max(-props.binEdges[0], props.binEdges[props.binEdges.length - 1])
      const colorScale = scaleLinear<string, string>()
          .domain([-maxV, 0, maxV])
          .range(props.colorScheme)

      const zeroScale = scaleLinear()
          .domain(extent(props.binEdges))
          .range([0, size.width - margin.l - margin.r])

      select(xAxis.value)
          .call(axisBottom(zeroScale).ticks(3))
      select(yAxis.value)
          .call(axisLeft(yScale))

      bars.value = props.values.map((value, i) => ({
        x: xScale(i),
        y: yScale(value),
        w: xScale(1) - xScale(0),
        h: size.height - margin.t - margin.b - yScale(value),
        label: `${value} [${props.binEdges[i]}, ${props.binEdges[i + 1]}]`,
        color: colorScale(.5 * (props.binEdges[i] + props.binEdges[i + 1]))
      } as BarRender))

    }

    watch(props => [props.values, props.binEdges], () => {
      updateVis()
    })

    onMounted(() => {
      updateVis()
    })

    let tippies = [];
    onUpdated(() => {
      tippies.forEach(t => t.destroy())
      const n = select(myself.value).selectAll('[data-tippy-content]').nodes()
      tippies = tippy(n as Element[])
    })


    return {size, margin, xAxis, yAxis, bars, myself}
  }
})
</script>

<style scoped>
.axis {
  font-size: 9pt;
  font-weight: inherit;
  color: #666;
}
</style>
