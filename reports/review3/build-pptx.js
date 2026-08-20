const pptxgen = require('pptxgenjs');
const path = require('path');
const html2pptx = require('./html2pptx');

async function main() {
  const pptx = new pptxgen();
  pptx.layout = 'LAYOUT_16x9';
  pptx.author = 'Huynh Quoc Trung';
  pptx.title = 'Vietnamese GraphRAG - Capstone Review 1';

  const slidesDir = path.join(__dirname, 'slides');
  const slideFiles = [
    'slide01-title.html',
    'slide02-problem.html',
    'slide03-users.html',
    'slide04-innovation.html',
    'slide05-architecture.html',
    'slide06-requirements.html',
    'slide07-ai-framing.html',
    'slide08-dataset.html',
    'slide09-related1.html',
    'slide10-related2.html',
    'slide11-related3.html',
    'slide12-pipeline.html',
    'slide13-results.html',
    'slide14-team.html',
    'slide15-timeline.html',
    'slide17-security-legal.html',
    'slide18-model-justification.html',
    'slide16-thanks.html',
  ];

  for (const file of slideFiles) {
    const htmlPath = path.join(slidesDir, file);
    console.log(`Processing ${file}...`);
    await html2pptx(htmlPath, pptx);
  }

  const outPath = path.join(__dirname, 'review1-presentation.pptx');
  await pptx.writeFile({ fileName: outPath });
  console.log(`\nPresentation saved: ${outPath}`);
}

main().catch(err => { console.error(err); process.exit(1); });
