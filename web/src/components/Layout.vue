<template>
	<div class="generate-workspace">
		<section class="generate-workspace__stage">
			<aside class="generate-workspace__panel">
				<div class="generate-workspace__panel-title">
					{{ props.panelTitle }}
				</div>
				<slot name="panel" />
			</aside>

			<div class="generate-workspace__content">
				<div class="generate-workspace__preview">
					<slot name="preview"> </slot>
				</div>
			</div>
		</section>
	</div>
</template>

<script lang="ts" setup>
defineOptions({ name: 'GenerateWorkspaceLayout' })

type Props = {
	panelTitle: string
}

const props = defineProps<Props>()
</script>

<style lang="scss">
@use '../styles/theme.scss' as *;

.generate-workspace {
	display: flex;
	flex-direction: column;
	height: 100%;
	min-height: 0;
	padding: 12px;
	width: 100%;

	.generate-workspace__stage {
		flex: 1;
		display: flex;
		min-height: 0;

		.generate-workspace__panel {
			width: 380px;
			background-color: $panel-bg;
			border: 1px solid $panel-border-color;
			margin-right: 12px;
			border-radius: $panel-border-radius-big;
			padding: 12px 16px;
			height: 100%;
			box-sizing: border-box;
			position: relative;
			min-height: 0;

			.generate-workspace__panel-title {
				font-size: 16px;
				height: 22px;
				line-height: 22px;
				color: $panel-title-color;
				margin-bottom: 12px;
				font-weight: 500;
				letter-spacing: 0px;
				vertical-align: middle;
			}
		}
		.generate-workspace__content {
			flex: 1;
			min-width: 0;
			height: 100%;
			display: flex;
			flex-direction: column;
			box-sizing: border-box;
			gap: 12px;
		}

		.generate-workspace__preview {
			background-color: $preview-bg;
			border: 1px solid $preview-border-color;
			border-radius: $preview-border-radius-big;
			flex: 1;
			min-height: 0;
		}
	}
}
</style>
