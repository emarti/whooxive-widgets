
from bokeh.plotting import figure, output_file, show
from bokeh.embed import components
from flask import render_template

# prepare some data
x = [1, 2, 3, 4, 5]
y = [6, 7, 2, 4, 5]

plot = figure(tools=TOOLS,
              title="test title")
plot.line(x, y, legend="Temperature", line_width=2)



script, div = components(plot)
return render_template('graph.html', script=script, div=div)

# output to static HTML file
# output_file("lines.html", title="line plot example")

# create a new plot with a title and axis labels
# p = figure(title="simple line example", x_axis_label='x', y_axis_label='y')

# add a line renderer with legend and line thickness
# p.line(x, y, legend="Temp.", line_width=2)

# show the results
# show(p)