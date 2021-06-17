<template>
  <div class="header">
    <div class="headertext">
      <span style="font-weight: bold;">LMdiff</span> - by the MIT-IBM AI Lab and
      Harvard NLP
      <div
          style="padding: 3pt;display: inline-block; vertical-align: baseline;cursor: pointer;"
          @click="$emit('aboutClicked')"
      >
        <svg class="question-icon" xmlns="http://www.w3.org/2000/svg"
             viewBox="0 0 32 32"><title>About</title>
          <path
              d="M16,2A14,14,0,1,0,30,16,14,14,0,0,0,16,2Zm0,26A12,12,0,1,1,28,16,12,12,0,0,1,16,28Z"/>
          <circle cx="16" cy="23.5" r="1.5"/>
          <path
              d="M17,8H15.5A4.49,4.49,0,0,0,11,12.5V13h2v-.5A2.5,2.5,0,0,1,15.5,10H17a2.5,2.5,0,0,1,0,5H15v4.5h2V17a4.5,4.5,0,0,0,0-9Z"/>
        </svg>
      </div>
    </div>
    <div class="headertext">
      <span
          style="background-color: #d6604d;padding: 2px 5px;white-space: nowrap;">
        <label for="m1_select">model1:</label>
        <!--         @update:modelValue="(e)=> console.log(e)"-->
        <select id="m1_select" :value="selectedM1"
                @input="emitValue('update:selectedM1', $event)"
        ><option
            v-for="model in allModels"
            :value="model.model"
            :key="model.model"
        >{{model.type}} -{{ model.model }}</option></select>
      </span>&nbsp;
      <span
          style="background-color: #4393c3;padding: 2px 5px;white-space: nowrap;">
        <label for="m2_select">model2:</label>
        <select id="m2_select" :value="selectedM2"
                @input="emitValue('update:selectedM2', $event)"
        ><option
            v-for="model in allModels"
            :value="model.model"
            :key="model.model"
        >{{model.type}} -{{ model.model }}</option></select>

      </span>

    </div>
  </div>
</template>
<script lang="ts">
import {defineComponent, PropType} from "@vue/runtime-core"

export default defineComponent({
  name: 'NavBar',
  props: {
    allModels: Array as PropType<any[]>,
    selectedM1: String,
    selectedM2: String
  },
  emits: ["update:selectedM1", "update:selectedM2", "aboutClicked"],
  setup(props,ctx){
    //todo: just a workaround for TS
    const emitValue = (signal, e) =>{
      ctx.emit(signal, e.target.value)
    }
    return{
      emitValue
    }

  }

})
</script>
<style scoped>
.question-icon {
  width: 1em;
  height: 1em;
  position: relative;
  top: .13em;
  stroke: #aaa;
  stroke-width: 2;
}

.question-icon:hover {
  stroke: white;
  stroke-width: 2.5;
}

.header {
  display: flex;
  flex-wrap: nowrap;
  align-items: center;
  justify-content: space-between;
  width: 100%;
  background-color: #2c2d4d;
  position: sticky;
  top: 0;
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
