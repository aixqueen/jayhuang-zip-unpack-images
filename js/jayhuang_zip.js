import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";

function buildViewURL(file) {
	const params = new URLSearchParams({
		filename: file.filename,
		type: file.type || "output",
	});
	if (file.subfolder) params.set("subfolder", file.subfolder);
	return api.apiURL("/view?" + params.toString());
}

async function uploadZipToInput(file) {
	const form = new FormData();
	form.append("image", file, file.name);
	form.append("type", "input");
	form.append("overwrite", "true");
	const resp = await api.fetchApi("/upload/image", {
		method: "POST",
		body: form,
	});
	if (!resp.ok) {
		throw new Error(`upload failed: ${resp.status}`);
	}
	return await resp.json();
}

app.registerExtension({
	name: "JAYHUANG.ZipNodes",
	async beforeRegisterNodeDef(nodeType, nodeData) {
		if (nodeData.name === "JAYHUANG_LoadImagesFromZip") {
			const onNodeCreated = nodeType.prototype.onNodeCreated;
			nodeType.prototype.onNodeCreated = function () {
				if (onNodeCreated) onNodeCreated.apply(this, arguments);

				const zipWidget = this.widgets?.find((w) => w.name === "zip_file");
				if (!zipWidget) return;
				zipWidget.label = "ZIP file";

				const uploadWidgets = (this.widgets || []).filter((w) => w?.type === "button" && w?.name === "Choose ZIP to upload");
				if (uploadWidgets.length > 0) {
					const keep = uploadWidgets[0];
					keep.__jayhuangZipUpload = true;
					this.widgets = (this.widgets || []).filter((w) => w === keep || !(w?.type === "button" && w?.name === "Choose ZIP to upload"));
				}

				let input = this._jayhuangZipUploadInput;
				if (!input) {
					input = document.createElement("input");
					input.type = "file";
					input.accept = ".zip,application/zip";
					input.style.display = "none";
					document.body.appendChild(input);
					this._jayhuangZipUploadInput = input;
				}

				input.onchange = async () => {
					try {
						const f = input.files?.[0];
						if (!f) return;
						const res = await uploadZipToInput(f);
						const uploadedPath = res.subfolder ? `${res.subfolder}/${res.name}` : res.name;
						if (zipWidget.options?.values && !zipWidget.options.values.includes(uploadedPath)) {
							zipWidget.options.values.push(uploadedPath);
							zipWidget.options.values.sort();
						}
						zipWidget.value = uploadedPath;
						app.graph.setDirtyCanvas(true, true);
					} catch (e) {
						console.error(e);
					} finally {
						input.value = "";
					}
				};

				let uploadButton = (this.widgets || []).find((w) => w?.__jayhuangZipUpload === true);
				if (!uploadButton) {
					uploadButton = this.addWidget("button", "Choose ZIP to upload", null, () => input.click(), { serialize: false });
					uploadButton.__jayhuangZipUpload = true;
				}
			};
		}

		if (nodeData.name === "JAYHUANG_SaveImagesToZip") {
			const onNodeCreated = nodeType.prototype.onNodeCreated;
			nodeType.prototype.onNodeCreated = function () {
				if (onNodeCreated) onNodeCreated.apply(this, arguments);

				const prefixWidget = this.widgets?.find((w) => w.name === "filename_prefix");
				if (prefixWidget) prefixWidget.label = "Filename prefix";
			};

			nodeType.prototype.onExecuted = function (message) {
				const file = message?.zip?.[0] || message?.images?.[0];
				if (!file) return;

				const url = buildViewURL(file);
				this._jayhuangZipDownloadUrl = url;

				const downloadWidgets = (this.widgets || []).filter((w) => w?.type === "button" && w?.name === "Download ZIP");
				if (downloadWidgets.length > 0) {
					const keep = downloadWidgets[0];
					keep.__jayhuangZipDownload = true;
					this.widgets = (this.widgets || []).filter((w) => w === keep || !(w?.type === "button" && w?.name === "Download ZIP"));
				}

				let downloadButton = (this.widgets || []).find((w) => w?.__jayhuangZipDownload === true);
				if (!downloadButton) {
					downloadButton = this.addWidget(
						"button",
						"Download ZIP",
						null,
						() => {
							const u = this._jayhuangZipDownloadUrl;
							if (u) window.open(u, "_blank", "noopener,noreferrer");
						},
						{ serialize: false }
					);
					downloadButton.__jayhuangZipDownload = true;
				}
				app.graph.setDirtyCanvas(true, true);
			};
		}
	},
});
