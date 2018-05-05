var system = require('system');
var args = system.args;
var fs = require('fs');
var cheerio = require("cheerio");
// 0 is node : 1 is the script path

var defs = cheerio.load( fs.readFileSync(process.argv[2]) )
var env = cheerio.load( fs.readFileSync(process.argv[3]) )

// First remove old defs
env("[href='#id_tag']").remove()
var newDefs = env("defs symbol");
newDefs.remove()

defs("*")
  .removeAttr("inkscape:label")
  .removeAttr("inkscape:connector-curvature");

defs("svg > g").each(function(idx, elm){
  
  var sym = env("<symbol id='"+defs(elm).attr("id")+"'>"+defs(elm).html()+"</symbol>");
  env("<title>"+defs(elm).attr("id")+"</title>").appendTo(sym)
  sym.appendTo( env("defs") )

})

env("text").attr("x", 50.0).attr("y", 35);

env('[href="#cose_MotionDetector"]').each(function(i, e){
  var sensor = env(e).siblings("text").text();
  if(sensor.substr(0,2) == 'MA'){
    env(e).attr("href","#cose_AreaMotionDetector")
  }

})

env("g > text").each(function(idx, elm){
  
  var g = env(elm.parentNode);
  
  var sensor = env(elm).text().toLowerCase();
  var envName = g.attr("id").split("_")[0];
  var use = g.find("use");
  var txt = g.find("text");

  g.attr("id", envName + "_" + sensor);
  txt.attr("id", envName + "_" + sensor + "_label");
  use.attr("id", envName + "_" + sensor + "_symbol");

})

console.log('<?xml version="1.0" encoding="UTF-8" standalone="no"?>')
console.log(env("body").html().replace(/href=/g,"href=") )/*xlink:*/
