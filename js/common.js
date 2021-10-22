"use strict";

// Common logic for Litmus pages
$(function () {

	window.common = {
		/*
		 * Choose API endpoint based on environment
		 */

		API_CODE: "Fn21WRKB0yuxmkRA68YTvpwXH2uu5dWIV1X/aVkdkOJCB8Ly/waauA==",

		 /* Safely encapsulates API calls and handling
		 * (Using function closures for maintaining state)
		 */
		getSafeSubmitHandler: function(realHandler, submitId){
			var inUse = false;
			function setInUse(){
				inUse = true;
				$("#"+submitId).prop('disabled', true);
			}
			function unsetInUse(){
				inUse = false;
				$("#"+submitId).prop('disabled', false);
			}
			return async function(){
				if (inUse) return;
				setInUse();
				realHandler(unsetInUse); // callback cleans up sessions
			}			
		},

		/*
		 * Reads file input data‭
		 * (Max input size (4MB) by default)
		 */
		readInputFileData: async function(fileInpId, maxSize=16000000){
			var files = document.getElementById(fileInpId).files;
			var data = null;
			if (files && files.length > 0 && files[0].size <= maxSize) {
				data = await new Response(files[0]).text();
			}   
			return data;
		},

		/*
		 * Wrapper for making AJAX POST requests
		 */
		ajaxPostRequest: function (args) {
			// Add your API code here args.params.code
			window.last_api_req = null;
			args.params.code = this.API_CODE;

			// Build XHR request
			var oReq = new XMLHttpRequest();
			oReq.open("POST", "https://litmus.azurewebsites.net/api/" + args.path + "?&" + $.param(args.params), true);
			oReq.responseType = "responseType" in args ? args.responseType : "json";

			// Build response handler
			oReq.onreadystatechange = function () {
				if(oReq.readyState === XMLHttpRequest.DONE) {
					window.last_api_req = oReq;
					var status = oReq.status;
					if (status >= 200 && status < 400) {
						args.onSuccess(oReq);
					} else {
						args.onError(oReq);
					}
				}
			};
			oReq.send(args.body);
		},

		/*
		 * Helper to save content to file
		 */
		downloadToFile: function(content, filename, contentType){
			const a = document.createElement('a');
			const file = new Blob([content], {type: contentType});

			a.href= URL.createObjectURL(file);
			a.download = filename;
			a.click();

			URL.revokeObjectURL(a.href);
		},

		/*
		 * Maps a JS object into a list 
		 * (optionally using a given transform)
		 */
		object2List: function(obj, mapper){
			if (typeof mapper == 'undefined')
				return Object.keys(obj).map(k => [k, obj[k]]);
			else
				return Object.keys(obj).map(k => [k, mapper(obj[k])]);
		},

		boundedCharter: function(series, series_avg, comparator, nbars, canvasId){
			
			// Preprocessing
			series.sort((a,b) => comparator(a[1],b[1]))
			if (series_avg == null)
				series_avg = series.reduce((acc,curr) => acc + curr[1], 0) / series.length;

			// Identify good and bad samples
			var top_n    = series.filter(x => comparator(x[1], series_avg) < 0).slice(0,nbars);
			var bottom_n = series.filter(x => comparator(x[1], series_avg) > 0).slice(-nbars);
			var showBars = top_n.concat([["Avg.", series_avg]], bottom_n);

			// Draw chart
			var ctx = document.getElementById(canvasId).getContext('2d');
			var chart = new Chart(ctx, {
			    type: 'horizontalBar',
			    data: {
			        labels: showBars.map(x => x[0]),
			        datasets: [{
			            data: showBars.map(x => x[1]),
			            backgroundColor: showBars.map(x => comparator(x[1], series_avg) < 0 ? "MediumSeaGreen" : (comparator(x[1], series_avg) == 0 ? "Orange" : "Tomato")),
			            hoverBackgroundColor: showBars.map(x => comparator(x[1], series_avg) < 0 ? "MediumSeaGreen" : (comparator(x[1], series_avg) == 0 ? "Orange" : "Tomato")) 
			        }]
			    },
			    options: {
			    	scales: {
						xAxes: [{
							display: true,
							ticks: {
								beginAtZero: true,
								suggestedMin: 0,
								suggestedMax: 100
							}
						}]
					},
					legend: {
						display: false
					}
			    }
			});
		},

		boundedCharterPie: function (series, modified_series, comparator, budget, canvasId) {		
			
			function getRandomColor(label) {

				if (label == "Unused"){
					return "#A9A9A9";
				}

				var letters = '0123456789ABCDEF'.split('');
				var color = '#';
				for (var i = 0; i < 6; i++) {
					color += letters[Math.floor(Math.random() * 16)];
				}

				if (color == "#A9A9A9") {
					color = '#';
					for (var i = 0; i < 6; i++) {
						color += letters[Math.floor(Math.random() * 16)];
					}
				}
				return color;

			}

			var data_sz = series.map(x => x[1]);
			var deno_sum = data_sz.reduce((a, b) => a + b, 0)
			var remaining_budget = budget - deno_sum;

			if (remaining_budget > 0) {
				modified_series.push(["Unused", Math.round(((remaining_budget * 1000) / (budget * 10))*100)/100]);
			}

			series.sort((a, b) => comparator(a[1], b[1]))
			modified_series.sort((a, b) => comparator(a[1], b[1]))
			// Draw Pie chart
			var ctx = document.getElementById(canvasId).getContext('2d');
			var chart = new Chart(ctx, {
				type: 'pie',
				data: {
					labels: modified_series.map(x => x[0]),
					datasets: [{
						data: modified_series.map(x => Math.round(x[1] * 100) / 100),
						backgroundColor: modified_series.map(x => getRandomColor(x[0])),
					}]
				}
			});
		},


	};
})