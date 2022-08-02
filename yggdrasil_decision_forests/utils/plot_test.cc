/*
 * Copyright 2021 Google LLC.
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

  multiplot.items.push_back({Plot()});
  multiplot.items.back().plot.items.push_back(absl::make_unique<Curve>());
  auto* curve =
      dynamic_cast<Curve*>(multiplot.items.back().plot.items.back().get());
  EXPECT_OK(multiplot.Check());

  multiplot.items.back().col = 1;
  EXPECT_OK(multiplot.Check());

  multiplot.items.back().col = 2;
  EXPECT_THAT(multiplot.Check(),
              test::StatusIs(absl::StatusCode::kInvalidArgument));

  multiplot.items.back().col = 1;
  multiplot.items.back().num_cols = 2;
  EXPECT_THAT(multiplot.Check(),
              test::StatusIs(absl::StatusCode::kInvalidArgument));

  multiplot.items.back().num_cols = 1;
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

  const auto html_plot = ExportToHtml(plot).value();
  const auto path = file::JoinPath(test::TmpDirectory(), "plot.html");
  LOG(INFO) << "path: " << path;
  CHECK_OK(file::SetContent(path, html_plot));

  // The plot has been checked by hand.
  EXPECT_EQ(html_plot, R"(
<link href="https://www.gstatic.com/external_hosted/c3/c3.min.css" rel="stylesheet">
<script src="https://d3js.org/d3.v3.min.js" charset="utf-8"></script>
<script src="https://www.gstatic.com/external_hosted/c3/c3.min.js"></script>
<div id="chard_1"></div>
<script>
var chard_1 = c3.generate({
  bindto: '#chard_1',
  data: {
      columns: [
['curve_0', 2,0.5,4],
['curve_0_x', 1,2,3],
      ],
      names: { curve_0: 'curve 1',
      },
      xs: {'curve_0': 'curve_0_x',
      },
  },
  axis: {
  },
title: { text: 'Hello world'},
});
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

    multiplot.items.push_back({std::move(plot), 0, 0});
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

    multiplot.items.push_back({std::move(plot), 1, 0});
  }

  {
    Plot plot;
    plot.chart_id = "chard_3";
    plot.title = "Plot 3";

    auto curve = absl::make_unique<Curve>();
    curve->label = "curve 3";
    curve->xs = {10, 11, 15};
    curve->ys = {7, 9, 2};
    plot.items.push_back(std::move(curve));

    multiplot.items.push_back({std::move(plot), 0, 1, 2, 1});
  }

  const auto html_plot = ExportToHtml(multiplot).value();
  const auto path = file::JoinPath(test::TmpDirectory(), "multiplot.html");
  LOG(INFO) << "path: " << path;
  CHECK_OK(file::SetContent(path, html_plot));

  // The plot has been checked by hand.
  EXPECT_EQ(
      html_plot,
      R"(<div style='display: grid; gap: 0px; grid-template-columns: repeat(2, 2fr);'><div style='grid-row:1 / 2; grid-column:2 / 1;'>
<link href="https://www.gstatic.com/external_hosted/c3/c3.min.css" rel="stylesheet">
<script src="https://d3js.org/d3.v3.min.js" charset="utf-8"></script>
<script src="https://www.gstatic.com/external_hosted/c3/c3.min.js"></script>
<div id="chard_1"></div>
<script>
var chard_1 = c3.generate({
  bindto: '#chard_1',
  data: {
      columns: [
['curve_0', 2,0.5,4],
['curve_0_x', 1,2,3],
      ],
      names: { curve_0: 'curve 1',
      },
      xs: {'curve_0': 'curve_0_x',
      },
  },
  axis: {
x: { label: { text: 'x axis', position: 'outer-center' } },
y: { label: { text: 'y axis', position: 'outer-middle' } },
  },
title: { text: 'Plot 1'},
});
</script>
</div><div style='grid-row:1 / 2; grid-column:3 / 2;'><div id="chard_2"></div>
<script>
var chard_2 = c3.generate({
  bindto: '#chard_2',
  data: {
      columns: [
['curve_0', 7,9,2],
['curve_0_x', 10,11,15],
      ],
      names: { curve_0: 'curve 2',
      },
      xs: {'curve_0': 'curve_0_x',
      },
  },
  axis: {
  },
title: { text: 'Plot 2'},
});
</script>
</div><div style='grid-row:2 / 3; grid-column:3 / 1;'><div id="chard_3"></div>
<script>
var chard_3 = c3.generate({
  bindto: '#chard_3',
  data: {
      columns: [
['curve_0', 7,9,2],
['curve_0_x', 10,11,15],
      ],
      names: { curve_0: 'curve 3',
      },
      xs: {'curve_0': 'curve_0_x',
      },
  },
  axis: {
  },
title: { text: 'Plot 3'},
});
</script>
</div></div>)");
}

}  // namespace
}  // namespace plot
}  // namespace utils
}  // namespace yggdrasil_decision_forests
