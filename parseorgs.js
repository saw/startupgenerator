const parse = require('csv-parse');
const fs = require('fs');
const parser = parse({
    delimiter: ','
});

const dupes = {};
const output = fs.createWriteStream('companies.txt');

parser.on('readable', function(){
  let record;
  while (record = this.read()) {
      let name = record[1];
      if (!dupes[name]) {
          output.write(`${record[1]}\n`);
          dupes[name] = true;
      }  else {
          process.stdout.write(`${record[1]}...`);
      }
  }
});

let fileStream = new fs.createReadStream('organizations.csv');
fileStream.pipe(parser);