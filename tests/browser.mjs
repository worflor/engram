import assert from 'node:assert/strict';
import fs from 'node:fs';
import path from 'node:path';
import {fileURLToPath, pathToFileURL} from 'node:url';
import {chromium} from 'playwright-core';

const root=path.resolve(path.dirname(fileURLToPath(import.meta.url)),'..');
const executablePath=process.env.BROWSER_EXE || [
  'C:/Program Files (x86)/Microsoft/Edge/Application/msedge.exe',
  'C:/Program Files/Google/Chrome/Application/chrome.exe',
  '/usr/bin/google-chrome','/usr/bin/chromium'
].find(p=>fs.existsSync(p));
assert.ok(executablePath,'Set BROWSER_EXE to an installed Chromium-family browser');
const browser=await chromium.launch({headless:true,executablePath});
try {
  for(const width of [1280,390,320]) {
    const page=await browser.newPage({viewport:{width,height:1000}});
    const errors=[];page.on('pageerror',e=>errors.push(e.message));
    await page.goto(pathToFileURL(path.join(root,'instrument/engram.html')).href);
    for(const backend of ['Rust','Python']) for(const dim of ['4','8']) for(const signal of ['Orbit','Damped orbit','Ramp','Phase change','Noise']) {
      await page.selectOption('#backend',backend);await page.selectOption('#dim',dim);await page.selectOption('#signal',signal);
      assert.ok(Number(await page.locator('#blockCount').textContent())>0);
      await page.selectOption('#axis',String(Number(dim)-1));
      await page.locator('#position').fill('100');await page.locator('#position').dispatchEvent('input');
      assert.match(await page.locator('#readout').textContent(),/^100 /);
      await page.locator('#blocks button').last().click();
      assert.match(await page.locator('#blockDetail').textContent(),/K =/);
      assert.equal(await page.evaluate(()=>document.documentElement.scrollWidth>innerWidth+1),false);
    }
    for(const id of ['csv','json','packet','compact']) {
      const [download]=await Promise.all([page.waitForEvent('download'),page.click('#'+id)]);
      const file=await download.path();assert.ok(fs.statSync(file).size>0);
      if(id==='packet')assert.deepEqual([...fs.readFileSync(file).subarray(0,3)],[69,78,4]);
      if(id==='compact')assert.deepEqual([...fs.readFileSync(file).subarray(0,3)],[69,78,4]);
    }
    await page.selectOption('#signal','Orbit');await page.selectOption('#dim','4');
    if(process.env.SCREENSHOT_DIR){fs.mkdirSync(process.env.SCREENSHOT_DIR,{recursive:true});await page.screenshot({path:path.join(process.env.SCREENSHOT_DIR,`engram-${width}.png`),fullPage:true});}
    assert.deepEqual(errors,[]);await page.close();
  }
  console.log('Offline instrument: 60 viewport/case combinations and 12 actual downloads passed');
} finally {await browser.close();}
