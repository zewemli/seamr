var system = require('system');
var args = system.args;
var fs = require('fs');
var page = require('webpage').create();

var html = "<html><script type='text/javascript'>" + 
              "var edges = " + fs.read(args[2]) + ";\n" +
              fs.read("./jquery-3.2.0.min.js") + 
              ";"+fs.read("./d3.v4.min.js") + 
              "</script><body>" + fs.read(args[1]) + "</body></html>";

page.setContent(html, "http://phantomjs.org");

page.onConsoleMessage = function(msg, lineNum, sourceId) {
  console.log('CONSOLE: ' + msg + ' (from line #' + lineNum + ' in "' + sourceId + '")');
};

var val = page.evaluate(function startPage(){
  var pos = [];

  var xy = function( iden ){
    var n = d3.select("#"+iden);
    if(n.size() == 1){
      var nb = n.node().getBoundingClientRect();

      return {
        "x": nb.left + (nb.width / 8),
        "y": nb.top + (nb.height/3)
      }
    
    }else{
      return null;
    }

  };
  
  var known_edges = edges.filter(function(e){

                var c1 = xy(e[0]);
                var c2 = xy(e[1]);

                return (e[0].indexOf("_T") == -1) && (e[1].indexOf("_T") == -1) && c1 && c2;

              });

  var env = known_edges[0][0].split("_")[0];
  var env_sensors = env + "_sensors";

  var lnsBase = d3.select('body > svg').append("g")
              .classed("lines", true);

  var lns = lnsBase.selectAll("line").data(known_edges);
  
  lns.enter()
    .append("line")
    .attr("x1", function(e){ return xy(e[0]).x; })
    .attr("y1", function(e){ return xy(e[0]).y; })
    .attr("x2", function(e){ return xy(e[1]).x; })
    .attr("y2", function(e){ return xy(e[1]).y; })
    .style("stroke","black")
    .style("stroke-width", "4px")
    .style("opacity", 0.2);

  var sensors_layer = d3.select("g#" + env_sensors).node();
  if(sensors_layer == null){
    sensors_layer = d3.select("svg #sensors").node();
  }
  var p = sensors_layer.parentNode;

  sensors_layer.parentNode.removeChild(sensors_layer)
  p.appendChild(sensors_layer);

  d3.selectAll("svg use").each(function(){
    var href = d3.select(this).attr("xlink:href").substring(1).split("_");
    
    if( href[0] == "cose" ){
      
      var sensor = null;
      if(this.id.indexOf(env) == 0){
        sensor = this.id.split("_")[1].toLowerCase();
      }

      console.log( JSON.stringify({"id": this.id,
      "ref": href[0] + ":" + href[1],
      "sensor": sensor,
      "rect": this.getBoundingClientRect() }) ); 
    }

  })

  return $("body").html();
  
});

/*console.log(val);*/
phantom.exit();

