<template>

  <svg :width="size.width" :height="size.height" class="MiniHisto">
    <g class="xAxis axis" ref="xAxis"
       :transform="`translate(${margin.l}, ${size.height-margin.b})`">

      <text class="xLabel" v-for="(l,i) in xLabels" :x="l.x"
            :y="l.y"
            style="text-anchor: middle;alignment-baseline: hanging;fill:currentColor;">{{
          l.label
        }}
      </text>
    </g>
    <g class="yAxis axis" ref="yAxis"
       :transform="`translate(${margin.l}, ${margin.t})`"></g>
    <g ref="mainPlot" :transform="`translate(${margin.l}, ${margin.t})`">
      <rect v-for="bar in bars" class="bar"
            :x="bar.x" :y="bar.y" :width="bar.w" :height="bar.h"
            :style="{fill:bar.color?bar.color:null}"
      ></rect>
    </g>
  </svg>

</template>

<script lang="ts">
import {axisBottom, axisLeft, max, scaleLinear, scaleOrdinal, select} from "d3";
import {
  defineComponent, onMounted,
  PropType,
  reactive,
  ref,
  watch,
  watchEffect
} from "vue";
import {probDiffColors} from "../etc/colors";

interface BarRender {
  x: number,
  y: number,
  w: number,
  h: number,
  color?: string,
  label: string
}

interface AxisLabel {
  x: number,
  y: number,
  label: string
}


export default defineComponent({
  name: "MiniHisto",
  props: {
    dataPoints: {
      type: Array as PropType<number[]>,
      default: []
    },
    lowerThreshold: {
      type: Number,
      default: -0.01
    },
    upperThreshold: {
      type: Number,
      default: 0.01
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

    const mainPlot = ref(null as SVGGElement);
    const xAxis = ref(null as SVGGElement);
    const yAxis = ref(null as SVGGElement);
    const bars = ref([] as BarRender[]);
    const xLabels = ref([] as AxisLabel[]);

    const updateVis = () => {
      const lT = props.lowerThreshold;
      const uT = props.upperThreshold;


      const count = [[], [], []]
      props.dataPoints.forEach(d => {
        if (d < lT) count[0].push(d);
        else if (d > uT) count[2].push(d);
        else count[1].push(d);
      })
      const labels = [
        `x<${lT}`, `${lT} <= x <= ${uT}`, `x>${uT} `
      ]
      const cl = count.map((c, i) => ({
        l: c.length,
        label: labels[i],
        color: props.colorScheme[i]
      }));
      if (cl[1].l == 0) cl.splice(1, 1);

      const yScale = scaleLinear()
          .domain([0, max(cl.map(cl => cl.l))])
          .range([size.height - margin.t - margin.b, 0])
      const xScale = scaleLinear()
          .domain([0, cl.length])
          .range([0, size.width - margin.l - margin.r])


      bars.value = cl.map((cl, i) => ({
        x: xScale(i),
        y: yScale(cl.l),
        w: xScale(1) - xScale(0),
        h: size.height - margin.t - margin.b - yScale(cl.l),
        label: `${cl.label} [${cl.l}]`,
        color: cl.color
      } as BarRender))
      console.log(xAxis.value, "--- xAxis.value");
      // select(xAxis.value).call(axisBottom(xScale));
      xLabels.value = cl.map((c, i) => ({
        x: xScale(i + .5),
        y: (i%2) * 10,
        label: c.label
      } as AxisLabel))
      select(yAxis.value).call(axisLeft(yScale));
    }

    watchEffect(() => {
      updateVis();
    })

    onMounted(() => {
      updateVis();
    })


    // watch(props => props.dataPoints,(dataPoints)=>{
    //
    //
    //
    // })

    return {
      size, margin, mainPlot, xAxis, yAxis, bars, xLabels
    }
  }
})
</script>

<style scoped>
.axis {
  font-size: 9pt;
  font-weight: inherit;
  color: #666;
}
.bar{
  stroke: #333333;
  stroke-width: 1;
}

</style>
