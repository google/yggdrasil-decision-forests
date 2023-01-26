/*
 * Copyright 2022 Google LLC.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     https://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "yggdrasil_decision_forests/utils/plot.h"

#include "gtest/gtest.h"
#include "absl/memory/memory.h"
#include "yggdrasil_decision_forests/utils/filesystem.h"
#include "yggdrasil_decision_forests/utils/test.h"

namespace yggdrasil_decision_forests {
namespace utils {
namespace plot {

namespace {

// Check the validity of the plot.
TEST(MultiPlot, Check) {
  MultiPlot multiplot;
  multiplot.num_cols = -1;
  multiplot.num_rows = 2;

  EXPECT_THAT(multiplot.Check(),
              test::StatusIs(absl::StatusCode::kInvalidArgument));

  multiplot.num_cols = 2;
  EXPECT_OK(multiplot.Check());

  multiplot.items.push_back(absl::make_unique<plot::MultiPlotItem>());
  multiplot.items.back()->plot.items.push_back(absl::make_unique<Curve>());
  auto* curve =
      dynamic_cast<Curve*>(multiplot.items.back()->plot.items.back().get());
  EXPECT_OK(multiplot.Check());

  multiplot.items.back()->col = 1;
  EXPECT_OK(multiplot.Check());

  multiplot.items.back()->col = 2;
  EXPECT_THAT(multiplot.Check(),
              test::StatusIs(absl::StatusCode::kInvalidArgument));

  multiplot.items.back()->col = 1;
  multiplot.items.back()->num_cols = 2;
  EXPECT_THAT(multiplot.Check(),
              test::StatusIs(absl::StatusCode::kInvalidArgument));

  multiplot.items.back()->num_cols = 1;
  curve->ys.push_back(1);
  curve->ys.push_back(2);
  EXPECT_OK(multiplot.Check());

  curve->xs.push_back(1);
  EXPECT_THAT(multiplot.Check(),
              test::StatusIs(absl::StatusCode::kInvalidArgument));

  curve->xs.push_back(2);
  EXPECT_OK(multiplot.Check());
}

// Basic plotting.
TEST(Plot, Base) {
  Plot plot;
  plot.title = "Hello world";
  plot.chart_id = "chard_1";

  auto curve = absl::make_unique<Curve>();
  curve->label = "curve 1";
  curve->xs = {1, 2, 3};
  curve->ys = {2, 0.5, 4};
  plot.items.push_back(std::move(curve));
  plot.x_axis.label = "x label";
  plot.x_axis.scale = plot::AxisScale::LOG;
  plot.x_axis.manual_tick_values = std::vector<double>();
  plot.x_axis.manual_tick_values->push_back(1);
  plot.x_axis.manual_tick_values->push_back(2);
  plot.x_axis.manual_tick_texts = std::vector<std::string>();
  plot.x_axis.manual_tick_texts->push_back("v1");
  plot.x_axis.manual_tick_texts->push_back("v2");

  const auto html_plot = ExportToHtml(plot).value();
  const auto path = file::JoinPath(test::TmpDirectory(), "plot.html");
  YDF_LOG(INFO) << "path: " << path;
  CHECK_OK(file::SetContent(path, html_plot));

  // The plot has been checked by hand.
  EXPECT_EQ(
      html_plot,
      R"(<script src='https://www.gstatic.com/external_hosted/plotly/plotly.min.js'></script>
<div id="chard_1" style="display: inline-block;" ></div>
<script>
  Plotly.newPlot(
    'chard_1',
    [{
x: [1,2,3],
y: [2,0.5,4],
type: 'scatter',
mode: 'lines',
line: {
  dash: 'solid',
  width: 1
},
name: 'curve 1',
},
],
    {
      width: 600,
      height: 400,
      title: 'Hello world',
      showlegend: true,
      xaxis: {
        ticks: 'outside',
        showgrid: true,
        zeroline: false,
        showline: true,
        title: 'x label', type: 'log',tickvals: [1,2],ticktext: ["v1","v2",],
        },
      font: {
        size: 10,
        },
      yaxis: {
        ticks: 'outside',
        showgrid: true,
        zeroline: false,
        showline: true,
        title: '',
        },
      margin: {
        l: 50,
        r: 50,
        b: 50,
        t: 50,
      },
    },
    {
      modeBarButtonsToRemove: ['sendDataToCloud'],
      displaylogo: false,displayModeBar: false,
    }
  );
</script>
)");
}

// Basic multi-plotting.
TEST(MultiPlot, Base) {
  MultiPlot multiplot;
  multiplot.num_cols = 2;
  multiplot.num_rows = 2;

  {
    Plot plot;
    plot.chart_id = "chard_1";
    plot.title = "Plot 1";
    plot.x_axis.label = "x axis";
    plot.y_axis.label = "y axis";

    auto curve = absl::make_unique<Curve>();
    curve->label = "curve 1";
    curve->xs = {1, 2, 3};
    curve->ys = {2, 0.5, 4};
    plot.items.push_back(std::move(curve));

    multiplot.items.push_back(
        absl::make_unique<plot::MultiPlotItem>(std::move(plot), 0, 0));
  }

  {
    Plot plot;
    plot.chart_id = "chard_2";
    plot.title = "Plot 2";

    auto curve = absl::make_unique<Curve>();
    curve->label = "curve 2";
    curve->xs = {10, 11, 15};
    curve->ys = {7, 9, 2};
    plot.items.push_back(std::move(curve));

    multiplot.items.push_back(
        absl::make_unique<plot::MultiPlotItem>(std::move(plot), 1, 0));
  }

  {
    Plot plot;
    plot.chart_id = "chard_3";
    plot.title = "Plot 3";

    auto bars = absl::make_unique<Bars>();
    bars->label = "bars 3";
    bars->centers = {10, 11, 15};
    bars->heights = {7, 9, 2};
    plot.items.push_back(std::move(bars));

    multiplot.items.push_back(
        absl::make_unique<plot::MultiPlotItem>(std::move(plot), 0, 1, 2, 1));
  }

  const auto html_plot = ExportToHtml(multiplot).value();
  const auto path = file::JoinPath(test::TmpDirectory(), "multiplot.html");
  YDF_LOG(INFO) << "path: " << path;
  CHECK_OK(file::SetContent(path, html_plot));

  // The plot has been checked by hand.
  EXPECT_EQ(
      html_plot,
      R"(<div style='display: grid; gap: 0px; grid-auto-columns: min-content;'><div style='grid-row:1 / span 1; grid-column:1 / span 1;'><script src='https://www.gstatic.com/external_hosted/plotly/plotly.min.js'></script>
<div id="chard_1" style="display: inline-block;" ></div>
<script>
  Plotly.newPlot(
    'chard_1',
    [{
x: [1,2,3],
y: [2,0.5,4],
type: 'scatter',
mode: 'lines',
line: {
  dash: 'solid',
  width: 1
},
name: 'curve 1',
},
],
    {
      width: 600,
      height: 400,
      title: 'Plot 1',
      showlegend: true,
      xaxis: {
        ticks: 'outside',
        showgrid: true,
        zeroline: false,
        showline: true,
        title: 'x axis',
        },
      font: {
        size: 10,
        },
      yaxis: {
        ticks: 'outside',
        showgrid: true,
        zeroline: false,
        showline: true,
        title: 'y axis',
        },
      margin: {
        l: 50,
        r: 50,
        b: 50,
        t: 50,
      },
    },
    {
      modeBarButtonsToRemove: ['sendDataToCloud'],
      displaylogo: false,displayModeBar: false,
    }
  );
</script>
</div><div style='grid-row:1 / span 1; grid-column:2 / span 1;'>
<div id="chard_2" style="display: inline-block;" ></div>
<script>
  Plotly.newPlot(
    'chard_2',
    [{
x: [10,11,15],
y: [7,9,2],
type: 'scatter',
mode: 'lines',
line: {
  dash: 'solid',
  width: 1
},
name: 'curve 2',
},
],
    {
      width: 600,
      height: 400,
      title: 'Plot 2',
      showlegend: true,
      xaxis: {
        ticks: 'outside',
        showgrid: true,
        zeroline: false,
        showline: true,
        title: '',
        },
      font: {
        size: 10,
        },
      yaxis: {
        ticks: 'outside',
        showgrid: true,
        zeroline: false,
        showline: true,
        title: '',
        },
      margin: {
        l: 50,
        r: 50,
        b: 50,
        t: 50,
      },
    },
    {
      modeBarButtonsToRemove: ['sendDataToCloud'],
      displaylogo: false,displayModeBar: false,
    }
  );
</script>
</div><div style='grid-row:2 / span 1; grid-column:1 / span 2;'>
<div id="chard_3" style="display: inline-block;" ></div>
<script>
  Plotly.newPlot(
    'chard_3',
    [{
x: [10,11,15],
y: [7,9,2],
type: 'bar',
name: 'bars 3',
},
],
    {
      width: 600,
      height: 400,
      title: 'Plot 3',
      showlegend: true,
      xaxis: {
        ticks: 'outside',
        showgrid: true,
        zeroline: false,
        showline: true,
        title: '',
        },
      font: {
        size: 10,
        },
      yaxis: {
        ticks: 'outside',
        showgrid: true,
        zeroline: false,
        showline: true,
        title: '',
        },
      margin: {
        l: 50,
        r: 50,
        b: 50,
        t: 50,
      },
    },
    {
      modeBarButtonsToRemove: ['sendDataToCloud'],
      displaylogo: false,displayModeBar: false,
    }
  );
</script>
</div></div>)");
}

TEST(Bars, FromHistogram) {
  Bars bars;
  const auto hist =
      utils::histogram::Histogram<float>::MakeUniform({1, 1, 5, 10, 11}, 3);
  CHECK_OK(bars.FromHistogram(hist));

  const float eps = 0.0001;
  EXPECT_EQ(bars.centers.size(), 3);
  EXPECT_NEAR(bars.centers[0], 2.83333, eps);
  EXPECT_NEAR(bars.centers[1], 6.5, eps);
  EXPECT_NEAR(bars.centers[2], 9.66667, eps);
  EXPECT_EQ(bars.heights.size(), 3);
  EXPECT_NEAR(bars.heights[0], 2, eps);
  EXPECT_NEAR(bars.heights[1], 1, eps);
  EXPECT_NEAR(bars.heights[2], 2, eps);
}

TEST(Bars, PlotPlacer) {
  MultiPlot multiplot;
  auto placer = PlotPlacer::Create(3, 2, &multiplot).value();
  auto* plot_1 = placer.NewPlot().value();
  auto* plot_2 = placer.NewPlot().value();
  auto* plot_3 = placer.NewPlot().value();
  CHECK_OK(placer.Finalize());

  EXPECT_EQ(multiplot.num_cols, 2);
  EXPECT_EQ(multiplot.num_rows, 2);
  EXPECT_EQ(multiplot.items.size(), 3);

  EXPECT_EQ(plot_1, &multiplot.items[0]->plot);
  EXPECT_EQ(multiplot.items[0]->col, 0);
  EXPECT_EQ(multiplot.items[0]->row, 0);

  EXPECT_EQ(plot_2, &multiplot.items[1]->plot);
  EXPECT_EQ(multiplot.items[1]->col, 1);
  EXPECT_EQ(multiplot.items[1]->row, 0);

  EXPECT_EQ(plot_3, &multiplot.items[2]->plot);
  EXPECT_EQ(multiplot.items[2]->col, 0);
  EXPECT_EQ(multiplot.items[2]->row, 1);
}

}  // namespace
}  // namespace plot
}  // namespace utils
}  // namespace yggdrasil_decision_forests
