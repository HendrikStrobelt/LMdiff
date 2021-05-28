<template>
  <div class="header">
    <div class="headertext">
      <span style="font-weight: bold;">LMdiff</span> - Diffs for
      Transformer language models
    </div>
    <div class="headertext">
      <span
          style="background-color: #d6604d;padding: 2px 5px;white-space: nowrap;">
        <label for="m1_select">model1:</label>
<!--         @update:modelValue="(e)=> console.log(e)"-->
        <select id="m1_select" :value="selectedM1"
          @input="$emit('update:selectedM1', $event.target.value)"
        ><option
            v-for="model in allModels"
            :value="model.model"
            :key="model.model"
        >{{ model.model }}</option></select>
      </span>&nbsp;
      <span
          style="background-color: #4393c3;padding: 2px 5px;white-space: nowrap;">
        <label for="m2_select">model2:</label>
        <select id="m2_select" :value="selectedM2"
                @input="$emit('update:selectedM2', $event.target.value)"
        ><option
            v-for="model in allModels"
            :value="model.model"
            :key="model.model"
        >{{ model.model }}</option></select>

      </span>
    </div>
  </div>
</template>
<script lang="ts">
import { defineComponent, PropType } from "@vue/runtime-core"

export default defineComponent({
  name: 'NavBar',
  props: {
    allModels: Array as PropType<any[]>,
    selectedM1: String,
    selectedM2: String
  },
  emits:["update:selectedM1","update:selectedM2" ]
})
</script>
<style>

.header {
  display: flex;
  flex-wrap: nowrap;
  align-items: center;
  justify-content: space-between;
  width: 100%;
  background-color: #2c2d4d;
  /*min-height: 30px;*/
}

.headertext {
  /*display: flex;*/
  /*vertical-align: middle;*/
  text-align: left;
  /*font-size: 20px;*/
  font-weight: 500;
  color: #ffffff;
  padding: 5px;
  /*font-family: 'Source Sans Pro', sans-serif;*/
}



</style>
