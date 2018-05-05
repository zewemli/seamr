var system = require('system');
var args = system.args;
var fs = require('fs');
var cheerio = require("cheerio");
// 0 is node : 1 is the script path

var $ = cheerio.load( fs.readFileSync(process.argv[2]) )

var typeCnt = {};

$("use").each(function(idx, elm){

  var link = $(elm).attr("xlink:href");
  if( !link ){ link = $(elm).attr("href"); }

  var nParts = link.substring(1).split("_");
  var n = nParts[1] || nParts;
  if(!typeCnt[n]){ typeCnt[n] = 0; }
  typeCnt[n] += 1;

  $(elm).attr("id", n + "-" + typeCnt[n])
})

$("g > text").each(function(idx, elm){
  
  var g = $(elm.parentNode);
  
  var sensor = $(elm).text().toLowerCase();
  var envName = g.attr("id").split("_")[0];

  g.attr("id", envName + "_" + sensor);
  g.find("text").attr("id", envName + "_" + sensor + "_label");
  g.find("use").attr("id", envName + "_" + sensor + "_symbol");

})

console.log('<?xml version="1.0" encoding="UTF-8" standalone="no"?>')
console.log( $("body").html() )
