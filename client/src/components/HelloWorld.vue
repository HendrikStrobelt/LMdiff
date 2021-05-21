<template>
  <h1>{{ msg }}</h1>

  <button @click="count++">count is: {{ count }}</button>
  <p>
    Edit
    <code>components/HelloWorld.vue</code> to test hot module replacement.
  </p>

  <hr/>

  <input type="text" v-model="name" placeholder="What's my name?"/>
  <button @click="sayHello">Say Hello</button>
  <button @click="sayGoodbye">Say Goodbye</button>

  <div v-if="showText">{{showText}}</div>

</template>

<script lang="ts">
import { ref, defineComponent } from 'vue'
import {API} from "../api"

export default defineComponent({
  name: 'HelloWorld',
  props: {
    msg: {
      type: String,
      required: true
    }
  },
  setup: () => {
    const count = ref(0)
    const name = ref("")
    const api = new API()
    const showText = ref("")

    function sayHello() {
      api.getAHi(name.value).then(r => {
        showText.value = r
      })
    }

    function sayGoodbye() {
      api.postABye(name.value).then(r => {
        showText.value = r
      })
    }

    return { count, name, sayHello, sayGoodbye, showText }
  }
})
</script>

<style scoped>
a {
  color: #42b983;
}

label {
  margin: 0 0.5em;
  font-weight: bold;
}

code {
  background-color: #eee;
  padding: 2px 4px;
  border-radius: 4px;
  color: #304455;
}
</style>
