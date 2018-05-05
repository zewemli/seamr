var system = require('system');
var args = system.args;
var fs = require('fs');
var page = require('webpage').create();

var env_name = args[2]

var html = "<html><script type='text/javascript'>" + 
	      "window.env = '"+env_name+"';" +
              "window.sensors = " + JSON.stringify(args.slice(3)) + ";\n" +
              fs.read("./jquery-3.2.0.min.js") + 
              ";"+fs.read("./d3.v4.min.js") + 
              "</script><body>" + fs.read(args[1]) + "</body></html>";

page.setContent(html, "http://phantomjs.org");

page.onConsoleMessage = function(msg, lineNum, sourceId) {
  console.log( msg );
};

var val = page.evaluate(function startPage(){


  function btwn(a, b1, b2) {
    if ((a >= b1) && (a <= b2)) { return true; }
    if ((a >= b2) && (a <= b1)) { return true; }
    return false;
  }

  function line_line_intersect(line1, line2) {
    var x1 = line1.x1, x2 = line1.x2, x3 = line2.x1, x4 = line2.x2;
    var y1 = line1.y1, y2 = line1.y2, y3 = line2.y1, y4 = line2.y2;
    var pt_denom = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4);
    var pt_x_num = (x1*y2 - y1*x2) * (x3 - x4) - (x1 - x2) * (x3*y4 - y3*x4);
    var pt_y_num = (x1*y2 - y1*x2) * (y3 - y4) - (y1 - y2) * (x3*y4 - y3*x4);
    if (pt_denom == 0) { return false; }
    else { 
      var pt = {'x': pt_x_num / pt_denom, 'y': pt_y_num / pt_denom}; 
      return (btwn(pt.x, x1, x2) && btwn(pt.y, y1, y2) && btwn(pt.x, x3, x4) && btwn(pt.y, y3, y4));
    }
  }

  function pathSegs(pathEl){
    var pathLength = pathEl.getTotalLength();
    var dlist = [];
    var step = Math.max(10, pathLength / 2500);

    for (var i=0; i < pathLength; i += step) {
      dlist.push( pathEl.getPointAtLength( i ) );
    }

    var lines = [];

    for (var i=1; i < dlist.length; i++) {
      var p1 = dlist[i-1];
      var p2 = dlist[i];

      lines.push( {"x1": p1.x, "x2": p2.x,
                   "y1": p1.y, "y2": p2.y});

    }

    return lines;
  }

  function path_line_intersections(line2) {

    for (var i=0; i < wallLines.length; i++) {
      if( line_line_intersect(wallLines[i], line2) ){
        return true;
      }
    }
    
    return false;
  }

  function isOK(link) {

    var lnkLine = {
      x1: link.source.x,
      y1: link.source.y,

      x2: link.target.x,
      y2: link.target.y
    };

    if( link.source.id && link.target.id ){
      return !path_line_intersections( lnkLine );
    }else{
      return false;
    }
  }

  function project(x, y, r1, r2, r3){ return x*r1 + y*r2 + r3; }

  function setPos(elm, obj, selector, sensor){

    var bbox = elm.getBoundingClientRect();

    var xform = d3.select(elm).attr("transform");

    if(xform == null){
      
      obj.x = (bbox.left + bbox.right) / 2;
      obj.y = (bbox.top + bbox.bottom) / 2;

      obj.left = bbox.left;
      obj.right = bbox.right;
      obj.top = bbox.top;
      obj.bottom = bbox.bottom;

    }else{
      
      var matNums = xform.substring("matrix(".length, xform.length - 1).split(",").map(parseFloat);

      var n = 0;
      var x = 0;
      var y = 0;

      var left = Infinity;
      var right = -Infinity;
      var top = Infinity;
      var bottom = -Infinity;

      d3.select(selector).selectAll("rect,path").each(function(){
        var bb = this.getBBox();

        var pos = [ [bb.x, bb.y],
                    [bb.x, bb.y + bb.height],
                    [bb.x + bb.width, bb.y],
                    [bb.x + bb.width, bb.y + bb.height] ];

        for(var i=0; i < pos.length; i++){
          n += 1;
          
          var nx = project(pos[i][0], pos[i][1], matNums[0], matNums[2], matNums[4]);
          var ny = project(pos[i][0], pos[i][1], matNums[1], matNums[3], matNums[5]);

          x += nx;
          y += ny; 
        
          top = Math.min(top, ny);
          bottom = Math.max(bottom, ny);
          left = Math.min(left, nx);
          right = Math.max(right, nx);

        }

      });

      obj.x = x / n;
      obj.y = y / n;

      obj.left = left;
      obj.right = right;
      obj.top = top;
      obj.bottom = bottom;
    
    }

    return obj;
  }

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

  var env_sensors = window.env + "_sensors";

  var sensors_layer = d3.select("g#" + env_sensors).node();
  if(sensors_layer == null){
    sensors_layer = d3.select("svg #sensors").node();
  }

  var sensorX = 0.0;
  var sensorY = 0.0;
  var p = sensors_layer.parentNode;
  var pageCnt = 0;

  var sLinks = [];
  var sites  = [];
  var sensor_sites = [];

  var walls = d3.selectAll("svg #walls path");
  var wallLines = [];

  var yMax = 0;
  var xMax = 0;  

  walls.each(function(){
    var r = this.getBoundingClientRect();
    xMax = Math.max(r.right, xMax);
    yMax = Math.max(r.bottom, yMax);
  })

  walls.each(function(){
    var segs = pathSegs(this);
    for(var i=0; i < segs.length; i++){
      wallLines.push(segs[i]);
    }
  });

  d3.selectAll("svg symbol").attr("transform", null);

  d3.selectAll("svg use").each(function(){
    var me = d3.select(this);
    var symRef = me.attr("href") || me.attr("xlink:href"); 
    var href = symRef.substring(1).split("_");

    if(!this.id){
      me.attr("id", "refTo"+pageCnt);
      pageCnt++;
    }

    if( (href[0] == "cose") ){
      
      if(this.id.indexOf(env) == 0){

        var sensor = this.parentNode.id.replace("_",":");

        var sensor_obj = setPos(d3.select(this.parentNode).selectAll("text").node(),
                                        {"id" : this.parentNode.id,
                                         "ref" : href[0] + ":" + href[1],
                                         "sensor" : sensor }, symRef, sensor);
        sites.push( sensor_obj );
        sensor_sites.push(sensor_obj);

      }else if( href[1] != "Window" ){
        sites.push( setPos(this, {"id": this.id,
                                  "ref": href[0] + ":" + href[1],
                                  "sensor": null}, symRef) );
      }
    }

  })

  var v = d3.voronoi()
            .x(function(d){ return d.x; })
            .y(function(d){ return d.y; });

  var links = [];
  
  for(var i=0; i < sites.length; i++){
    for(var j=i+1; j < sites.length; j++){
      /* -- */
      var edge = {
        "source": sites[i],
        "target": sites[j]
      }

      if( isOK(edge) ){
        links.push( edge );
      }
      /* -- */
    }
  }
  
  links.forEach(function( link ){
    sLinks.push( [ link.source.id, link.target.id ] );
 });

  var lines = d3.select("svg")
                .selectAll("line.delauney")
                .data( links );
  
  lines.enter()
    .append("line")
    .classed("delauney",true)
    .attr("x1", function(d){ return d.source.x; })
    .attr("y1", function(d){ return d.source.y; })
    .attr("x2", function(d){ return d.target.x; })
    .attr("y2", function(d){ return d.target.y; })
    .style("stroke", "rgba(0,0,0,0.3)")
    .style("stroke-width", "6")

  /*
  Draw red line for each segment. Good for debugging.
  d3.select("svg")
    .selectAll("line.ghl")
    .data( wallLines )
    .enter()
      .append("line")
      .classed("ghl", true)
      .attr("x1", function(d){ return d.x1; })
      .attr("y1", function(d){ return d.y1; })
      .attr("x2", function(d){ return d.x2; })
      .attr("y2", function(d){ return d.y2; })
      .style("stroke", "rgba(255,0,0,0.3)")
      .style("stroke-width", "4")
  */

  sensors_layer.parentNode.removeChild(sensors_layer)
  p.appendChild(sensors_layer);
  
  var gsize = 10;
  var grid = [];
  var x = 0;
  var xn = 0;

  while( x < xMax ){
    var y = 0;
    var yn = 0;

    while( y < yMax ){

      var lns = [{
        x1: x, x2: x,
        y1: y, y2: y + gsize
      },
      {
        x1: x + gsize, x2: x + gsize,
        y1: y, y2: y + gsize
      },
      {
        x1: x, x2: x + gsize,
        y1: y, y2: y
      },
      {
        x1: x, x2: x + gsize,
        y1: y + gsize, y2: y + gsize
      }]

      var ok = lns.reduce(function(so_far, l){ return so_far || path_line_intersections( l ); }, false);

      grid.push( { "x":x,
                   "y":yMax - y,
                   "xmax": x + gsize,
                   "ymax": Math.max(0.0, yMax - (y + gsize)),
                   "xn": xn,
                   "yn": yn,
                   "obs": 0 + ok } )

      y = y + gsize;
      yn += 1;
    }

    x = x + gsize;
    xn += 1;
  }

  return {"graph": { 'edges':sLinks, 'nodes': sites.filter(function(v){ return v.id != undefined; }) },
          "layout": $("body").html(),
          "grid": grid
        };
  
});

fs.write("../graphs/"+env_name+".json", JSON.stringify(val.graph, null, 4));
fs.write("../grids/"+env_name+".json", JSON.stringify(val.grid, null, 4));
fs.write("../plots/"+env_name+"_delauney.svg", val.layout);

phantom.exit();

