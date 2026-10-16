import assert from 'node:assert/strict';
import fs from 'node:fs';
import path from 'node:path';
import {spawn} from 'node:child_process';
import {fileURLToPath} from 'node:url';
import {chromium} from 'playwright-core';

const root=path.resolve(path.dirname(fileURLToPath(import.meta.url)),'..');
const executablePath=process.env.BROWSER_EXE || ['C:/Program Files (x86)/Microsoft/Edge/Application/msedge.exe','C:/Program Files/Google/Chrome/Application/chrome.exe','/usr/bin/google-chrome','/usr/bin/chromium'].find(p=>fs.existsSync(p));
const python=process.env.PYTHON_EXE || (fs.existsSync(path.join(root,'.venv/Scripts/python.exe'))?path.join(root,'.venv/Scripts/python.exe'):'python');
const server=spawn(python,['tools/serve_lab.py','--port','0'],{cwd:root,stdio:['ignore','pipe','pipe'],windowsHide:true});
let stderr='';server.stderr.on('data',chunk=>stderr+=chunk);
let browser;
try{
  const url=await new Promise((resolve,reject)=>{const timeout=setTimeout(()=>reject(Error('Lab did not start: '+stderr)),15000);server.once('exit',code=>{clearTimeout(timeout);reject(Error('Lab exited '+code+': '+stderr));});server.stdout.on('data',chunk=>{const match=String(chunk).match(/http:\/\/127\.0\.0\.1:\d+/);if(match){clearTimeout(timeout);resolve(match[0]);}});});
  browser=await chromium.launch({headless:true,executablePath});
  for(const width of [1280,390,320]){
    const page=await browser.newPage({viewport:{width,height:1000}});const errors=[];page.on('pageerror',error=>errors.push(error.message));
    await page.goto(url);
    await page.click('#train');await page.waitForFunction(()=>document.querySelector('#status').textContent.startsWith('Ready'));
    assert.equal(await page.locator('#nodes').textContent(),'3');assert.equal(await page.locator('#accuracy').textContent(),'100.0%');
    assert.equal(await page.locator('#network circle').count(),3);
    await page.selectOption('#family','2');await page.fill('#probeSeed','456');await page.click('#probe');await page.waitForFunction(()=>document.querySelector('#prediction').textContent.startsWith('rapid orbit'));
    await page.fill('#walk','4');await page.fill('#dreamNoise','0.5');await page.check('#stabilize');await page.click('#dream');await page.waitForFunction(()=>document.querySelector('#prediction').textContent.includes(' → '));
    await page.waitForFunction(()=>document.querySelector('#status').textContent.startsWith('Ready'));
    for(const id of ['session','checkpoint']){const[download]=await Promise.all([page.waitForEvent('download'),page.click('#'+id)]);const bytes=fs.readFileSync(await download.path());assert.ok(bytes.length>100);if(id==='session'){const state=JSON.parse(bytes);assert.equal(state.current.itinerary.length,6);assert.equal(state.report.held_out_observations,60);}else assert.equal(bytes.subarray(0,2).toString(),'PK');}
    assert.equal(await page.evaluate(()=>document.documentElement.scrollWidth>innerWidth+1),false);
    assert.deepEqual(errors,[]);
    if(process.env.SCREENSHOT_DIR){fs.mkdirSync(process.env.SCREENSHOT_DIR,{recursive:true});await page.screenshot({path:path.join(process.env.SCREENSHOT_DIR,`wells-${width}.png`),fullPage:true});}
    await page.close();
  }
  console.log('Live well lab: training, held-out probing, graph dreams and 6 verified downloads passed at 3 widths');
}finally{if(browser)await browser.close();server.kill();}
