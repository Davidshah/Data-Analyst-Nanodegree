# Data Visualization: Titanic Survival Data

### Summary
On April 15, 1912, the ![RMS Titanic](https://en.wikipedia.org/wiki/RMS_Titanic) collided with an iceberg and sank in the North Atlantic Ocean. Of the 2,224 passengers and crew aboard, less than 725 survived. This project will graphically portray the different survival chances based on gender and class. Females clearly have a higher survival rate than males and higher classes have a higher survival rate than lower classes.

### Design
v1. A simple bar chart used to show the survival rates of passengers (calculated in R) categorized by class and sex. A bar chart was selected for its efficiency is displaying categorical data. Class data is encoded along the X axis and Sex data is encoded through color.

v2. Axis labels clarified. Y axis modified to display % instead decimal. Tooltip corrected to accurately display survival rates.

v3. Gridlines removed in order to clean up the graphic.

v4. Dimensions optimized in order to display graphic accurately across different window sizes.

### Feedback
Feedback One:
> The axis labels are too small. "Pclass" isn't an intuitive label. I'm also having problems with the size of the graph in my browser.

Feedback Two:
> I feel like the graph could be cleaner. Try removing the axis lines. Also, the graphic that pops up when you hover over a bar seems to be displaying the wrong survival rate. The title and y label seem to disappear depending on the size of my window.

Feedback three:
> I think percentages would look better than decimals on the Y axis. The sizing of the graphic is all over the place, maybe there is an issue with your javascript. I do like how clear the findings are. The survival rate of women in first class is very surprising.

### Resources
* https://www.kaggle.com/c/titanic
* http://dimplejs.org/examples_index.html
* http://dimplejs.org/advanced_examples_viewer.html?id=advanced_custom_styling
* https://github.com/PMSI-AlignAlytics/dimple/wiki/dimple.legend
* http://stackoverflow.com/questions/10201841/display-y-axis-as-percentages
* http://stackoverflow.com/questions/3437786/get-the-size-of-the-screen-current-web-page-and-browser-window
