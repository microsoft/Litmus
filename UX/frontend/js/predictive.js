"use strict";

$(function () {
	var Common = window.common;

	/*
	 * Form UX handlers
	 */
	// Supported set of pre-existing (model, tasks)
	var pre_existing_sets = [
		["xlmr", "data_xlmr_xnli", "Classification (XLM-R + XNLI)"],
		["xlmr", "data_xlmr_wikiann", "Sequence Tagging (XLM-R + WikiAnn)"],
		["xlmr", "data_xlmr_udpos", "Sequence Tagging (XLM-R + UDPOS)"],
		["mbert", "custom", "Custom"],
		["xlmr", "custom", "Custom"]
	];

	// Form modifier for model-wise pre-existing datasets
	$("#model-select").on("change", function(){
		var options = "";
		pre_existing_sets.forEach(function(el){
			if (el[0] == $("#model-select").val())
				options += `<option value="${el[1]}">${el[2]}</option>`;
		})
		$("#existing-select").html(options).trigger('change');
	});

	// Form modifier for custom / pre-existing datasets
	$("#existing-select").on("change", function(){
		if ($("#existing-select").val() == "custom") {
			$('#train-data-info').show();
		}
		else {
			$('#train-data-info').hide();
		}
	});

	// Pre-populate the lang-specific budgets
	var updateLangSpecificBudgets = function(){
		var langs = $("#scoped-augmentable").val().split(",");
		var langBudgetString = "";
		for (var idx=0; idx<langs.length; idx++)
			if (langs[idx].trim() !== "")
				langBudgetString += langs[idx].trim() + ":" + $("#scoped-budget").val() + ",";
		
		if(langBudgetString !== "")
			$("#lang-specific-budget").val(langBudgetString.slice(0,-1));
	};
	$("#scoped-budget, #scoped-augmentable").on("change", updateLangSpecificBudgets);

	// Default augmentable langs to selected targets
	$("#scoped-targets").on("change", function(){
		$("#scoped-augmentable").val($("#scoped-targets").val());
		updateLangSpecificBudgets();
	});

	$(function () {
		$(".tooltipIcon").tooltip();
    });


    $('#if_suggestions').change(function () {
        if (this.checked) {
		$('#suggestions_inp').fadeIn('slow');
            predictive_mode = "2";
            $("#scoped-augmentable").val($("#scoped-targets").val());
            updateLangSpecificBudgets();

        }
        else {
		$('#suggestions_inp').fadeOut('slow');
            predictive_mode = "1";
            $('#scoped-pivots-row').val('');
            $('#scoped-budget').val(10000);
            $("#lang-specific-budget").val('');
            $('#suggestions_weights').val('');
            $('#suggestions_minperf').val('');
            $('#suggestions_minlangperf').val('');
            $("#inlineRadio1").prop("checked", true);
            $("#scoped-augmentable").val('');
        }
    });

    $('#edit-training-data').on('click', function () {
		$('#training_data_modal').modal('show');
    });

    $('#edit-budget').on('click', function () {
		$('#budget_modal').modal('show');
    });

    $('#edit-obj').on('click', function () {
		$('#objective_modal').modal('show');
    });

    $('#edit-perf').on('click', function () {
		$('#performance_modal').modal('show');
    });


    // Form modifier for mode dependent inputs
    $('input[name="objective-function"]').on("change", function () {
        if ($('input[name=objective-function]:checked', '#objective_modal').val() == "min") {
		$("#suggestions_weights").attr('disabled', 'disabled');
        }
        else {
		$("#suggestions_weights").removeAttr('disabled');
        }
	});

	"use strict";

	///////////////////////////////// Data placeholders /////////////////////////////////
	var validStrings = ['af', 'am', 'ar', 'as', 'az', 'be', 'bg', 'bn', 'br', 'bs', 'ca', 'cs', 'cy', 'da', 'de', 'el', 'en', 'es', 'et', 'eu', 'fa', 'fi', 'fr', 'fy', 'ga', 'gd', 'gl', 'gu', 'ha', 'he', 'hi', 'hu', 'hy', 'id', 'is', 'it', 'ja', 'jv', 'ka', 'kk', 'km', 'kn', 'ko', 'ku', 'ky', 'lo', 'lt', 'lv', 'mg', 'mk', 'ml', 'mr', 'ms', 'my', 'ne', 'nl', 'no', 'om', 'or', 'pa', 'pl', 'ps', 'pt', 'ro', 'ru', 'sd', 'si', 'sk', 'sl', 'so', 'sq', 'su', 'sv', 'sw', 'ta', 'te', 'th', 'tl', 'tr', 'ug', 'uk', 'ur', 'uz', 'vi', 'xh', 'yi', 'zh'];
	var currentRow = {
		strings: [],
		numbers: []
	};
	var predictive_mode = "1";

	///////////////////////////////// Utility functions /////////////////////////////////
	function sanitize(string) {
		const map = {
			'&': '&amp;',
			'<': '&lt;',
			'>': '&gt;',
			'"': '&quot;',
			"'": '&#x27;',
			"/": '&#x2F;',
		};
		const reg = /[&<>"'/]/ig;
		return string.replace(reg, (match) => (map[match]));
	}

	function initTestForm() {
		validStrings.forEach(e => {
			$("#validStrings").append(`<option value="${e}">${e}</option>`);
		});

		currentRow = {
			strings: [],
			numbers: []
		};
	}

	function renderRow(row) {
		var html = [];
		for (var i = 0; i < row.strings.length; i++) {
			html.push(`${row.strings[i]}(${row.numbers[i]})`);
		}
		html = html.join(", ");
		return html;
	}

	function getRowKey(key) {
		return `selectedRow_${key}`;
	}

	// add the current row display element
	function insertConfigRowHtml(rowVal) {
		var rowId = $(".config-row").length;
		var elementId = getRowKey(rowId);
		$("#formArea").append(`
        <div id="${elementId}" class="form-group input-group input-group-sm col-10">
            <div class="input-group-prepend">
                <div class="input-group-text form-control form-control-sm"><b>Config-${rowId + 1}</b></div>
            </div>
            <input type="text" class="form-control form-control-sm config-row" name="ConfigRow-${rowId}" id="ConfigRow-${rowId}" value="${rowVal}" disabled>
        </div>
        <button id="${elementId}Btn" type="button" class="btn btn-danger btn-sm col-2 align-self-start" title="Delete Configuration" onclick="window.deleteSavedRow(${rowId})">X</button>
    `);
		updateConfigsDropDown();
	}

	function deleteConfigRowHtml(rowId) {
		var rowElementId = getRowKey(rowId);
		var valStr = sanitize($(`#ConfigRow-${rowId}`).val());
		$(`#${rowElementId}`).remove();
		$(`#${rowElementId}Btn`).remove();
		return valStr;
	}

	window.deleteSavedRow = function (rowId) {
		var numRows = $(".config-row").length;

		deleteConfigRowHtml(rowId);
		if (numRows === 1) {
			$("#emptyConfigPlaceholder").css("display", "block");
			return;
		}

		// Deleting and re-inserting later rows to update their idx-dependent attributes
		var regenRows = [];
		for (var idx = rowId + 1; idx < numRows; idx++) {
			regenRows.push(deleteConfigRowHtml(idx));
		}

		for (var idx = 0; idx < regenRows.length; idx++) {
			insertConfigRowHtml(regenRows[idx]);
		}

		updateConfigsDropDown();
	}

	function updateConfigsDropDown() {
		var $el = $("#scoped-pivots-row");
		$el.empty();

		$el.append($("<option></option>").attr("value", "").text("Best of selected configs"));
		for (var idx = 0; idx < $(".config-row").length; idx++)
			$el.append($("<option></option>").attr("value", idx).text("Config-" + (idx + 1)));
	}

	///////////////////////////////// Adding event listeners /////////////////////////////////

	$("#addRowToSelected").click(() => {
		var selectedString = $("#validStrings option:selected").val();
		var selectedInputNumber = logsl.value(+$("#validInp").val()).toFixed(0);
		var pos = currentRow.strings.indexOf(selectedString);
		if (pos > -1) {
			currentRow.numbers[pos] = selectedInputNumber;
		}
		else {
			currentRow.strings.push(selectedString);
			currentRow.numbers.push(selectedInputNumber);
		}
		$("#selectedValidStrings").val(renderRow(currentRow));

	});

	$("#saveSelectedRow").click(() => {
		if (currentRow.strings.length === 0) {
			return;
		}

		$("#emptyConfigPlaceholder").css("display", "none");
		insertConfigRowHtml(sanitize($("#selectedValidStrings").val()));

		$("#selectedValidStrings").val("");
		currentRow = {
			strings: [],
			numbers: []
		};
	});

	function LogSlider(options) {
		options = options || {};
		this.minpos = options.minpos || 0;
		this.maxpos = options.maxpos || 100;
		this.minlval = Math.log(options.minval || 1);
		this.maxlval = Math.log(options.maxval || 100000);

		this.scale = (this.maxlval - this.minlval) / (this.maxpos - this.minpos);
	}

	LogSlider.prototype = {
		value: function (position) {
			return Math.exp((position - this.minpos) * this.scale + this.minlval);
		},
		position: function (value) {
			return this.minpos + (Math.log(value) - this.minlval) / this.scale;
		}
	};


	var logsl = new LogSlider({ maxpos: 100, minval: 100, maxval: 10000000 });

	$('#validInp').on('change', function () {
		var val = logsl.value(+$(this).val());
		$("#validInpLabel").html(`Task data size for pivot languages: ${val.toFixed(0)}`);
	});

	$('#value').on('keyup', function () {
		var pos = logsl.position(+$(this).val());
		$('#validInp').val(pos);
	});


	/*
	 * Transaction handlers
	 */

	// Update page to display results
	var showOutput = function(response){
		$("#output_div").css("display", "block");

		// helpers
		var prettyFloat = function(f) { return Math.round(f*1000)/10 + "%"; }

		var suggestions_avg = 0;
		var best_tgt_perfs = response["user-config-perfs"]["best-tgt-perfs"];
		var no_of_targets = response["user-config-perfs"]["best-tgt-perfs"].length;
		for (var i = 0; i < no_of_targets; i++) {
			suggestions_avg += best_tgt_perfs[i][1];
		}
		var avg = suggestions_avg / no_of_targets;

		// Show errors
		$("#reg_errors").html("Our predictions have a Mean Absolute Error of <b>" + prettyFloat(response.error) + "</b>.");
		
		// Show heatmap
		if (response?.heatmap) {
			// Display best row idx
			$("#best_row_idx").html(response["user-config-perfs"]["best-config-idx"]+1);
			
			// Draw tgt-perf prediction chart
			$("#chartContainer").html('<canvas id="targetPerfChart"></canvas>');
			Common.boundedCharter(
				response["user-config-perfs"]["best-tgt-perfs"].map(x => [x[0], Math.round(1000 * x[1]) / 10]),
				Math.round(1000 * avg)/10, 
				(a, b) => b - a, 5, 'targetPerfChart');

			// Display tgt-perf heatmap
			$("#heatmap_view").css("display", "block");
			$("#heatmap").css("display", "block");
			$("#heatmap").attr("src", "data:image/png;base64," + response.heatmap);
		}

		// Show PieChart
		if (response?.suggestions) {
			// Draw tgt-perf prediction chart
			$("#suggestions_table").css("display", "block");
			var baseline_accuracy_diff = Math.round((response.suggestions["search-perfs"]["augmented-perf"] - response.suggestions["search-perfs"]["baseline-perf"]) * 10000) / 100;
			var equal_aug_diff = Math.round((response.suggestions["search-perfs"]["augmented-perf"] - response.suggestions["search-perfs"]["equal-aug-perf"]) * 10000) / 100;

			if (baseline_accuracy_diff < 0.5 || response.suggestions["augments"].length == 0) {
				var no_results = "No data collection strategy found to improve";
				$("#diff_accuracy").html(no_results);
				$("#chart_title").html('');
			}
			else {
				$("#chart_title").css("display", "block");
				$("#suggestions_idx").html(response.suggestions["suggestions_row"]+1);
				
				$("#suggestionsChart").css("display", "block");
				$("#suggestionsChart").html('<canvas id="suggestionsPieChart"></canvas>');
				Common.boundedCharterPie(
					response.suggestions["augments"], response.suggestions["augments"].map(x => [x[0], Math.round(1000 * x[1]) / (10 * $("#scoped-budget").val())]), (a, b) => a - b, $("#scoped-budget").val(),
					'suggestionsPieChart');

				var accuracy_diff_html = "Outperforms baseline accuracy by " + baseline_accuracy_diff + " %"
				if (equal_aug_diff > 0)
					accuracy_diff_html += "<br>" + "Outperforms equal aug (target langs) accuracy by " + equal_aug_diff + "%";

				$("#diff_accuracy").html(accuracy_diff_html);
			}

		}
		
	};


	///////////////////////////////// Invoking startup functions /////////////////////////////////
	initTestForm();

	// Handler for predictive tool
	async function predictiveHandler(callback){
		// Clear outputs
		$("#error_div").css("display", "none");
		$("#output_div").css("display", "none");
		//$("#scoped-pivots-sizes").val("");
		$("#heatmap").css("display", "none");
		$("#suggestions_table").css("display", "none");
		$("#chart_title").css("display", "none");
		$("#heatmap_view").css("display", "none");
		$("#suggestionsChart").css("display", "none");

		// Make API call
		Common.ajaxPostRequest({
			path: "Predictor",
			params: {
				training_algorithm: "xgboost",
				model: $("#model-select").val(),
				pretrained: $("#existing-select").val(),
				predictive_mode: predictive_mode,
				pivots_and_sizes: JSON.stringify( $(".config-row").map( function(){ return $(this).val() }).get() ),
				suggestions_targets: $("#scoped-targets").val(),
				suggestions_augmentable: $("#scoped-augmentable").val(),
				suggestions_budget: $("#scoped-budget").val(),
				suggestions_pivots_row: $("#scoped-pivots-row").val(),
				suggestions_objective: $('input[name="objective-function"]:checked').val(),
				suggestions_lang_spec_budget: $("#lang-specific-budget").val(),
				suggestions_weights: $("#suggestions_weights").val(),
				suggestions_minperf: $("#suggestions_minperf").val(),
				suggestions_minlangperf: $("#suggestions_minlangperf").val()
			},
			body: $("#existing-select").val() == "custom" ? await Common.readInputFileData("train-data") : null,
			onSuccess: function(oReq){
				showOutput(oReq.response);
				callback();
			},
			onError: function(oReq){
				$("#error_div").css("display", "block");
				$("#error_div").html("<h2>Error Status: " + status + "</h2><br><p>" + oReq?.response?.error + "</p>");
				callback();
			}
		});
	}

	// Attach handler for querying API
	$("#submit").on("click", Common.getSafeSubmitHandler(predictiveHandler, "submit"));

	// Generic-data download handler
	$("#download").on("click", function(){
		var jsonData = JSON.stringify(window.last_api_req?.response, null, 4);
		Common.downloadToFile(jsonData, "litmus_results.json", "text/plain");
	});
});