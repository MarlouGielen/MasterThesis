<script>
	import Sidebar from './components/Sidebar.svelte';
	import Title from './components/Title.svelte';
	import SchematicCode from './components/SchematicCode.svelte';
	import SearchBar from './components/SearchBar.svelte';
	import Statistics from './components/Statistics.svelte';
	import CodeA from './components/CodeA.svelte';
	import CodeB from './components/CodeB.svelte';
  </script>

<style>
	.container {
		display: grid;
		grid-template-areas: 
		"title title title"
		"sidebar schematic-code search-bar"
		"sidebar schematic-code statistics"
		"sidebar schematic-code codeAB";
		grid-template-rows: 30px 50px 1fr 1.5fr; /* Adjust the heights of rows */
		grid-template-columns: 30px 1.25fr 1fr; /* Adjust the widths of columns */
		height: 100vh;
		gap: 10px; /* Optional: add some space between the grid items */
		border-width: 3px;
		border-style: solid;
		border-color: #83bdaa;
		background-color: #a5c7bc;
	}

</style>
  
<div class="container">
	<Title class="title" />
	<Sidebar class="sidebar" />
	<SchematicCode class="schematic-code" />
	<SearchBar class="search-bar" />
	<Statistics class="statistics" />
	<div style="grid-area: codeAB; display: flex;">
		<CodeA class="codeA" style="flex: 50%;" />
		<CodeB class="codeB" style="flex: 50%;" />
	</div>
</div>