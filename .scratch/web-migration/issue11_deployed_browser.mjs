import assert from 'node:assert/strict';
import {readFile} from 'node:fs/promises';
const root=process.env.AUDITORY_CHECK_ROOT;
assert(root);
const {chromium}=await import(root+'/node_modules/playwright/index.mjs');
const native=JSON.parse(await readFile('.scratch/web-migration/issue11-native.json','utf8'));
assert(native.completed);
const browser=await chromium.launch({headless:true,args:['--disable-gpu','--no-sandbox','--disable-dev-shm-usage','--no-zygote','--single-process'],env:{...process.env,LD_LIBRARY_PATH:''}});
try {
 const page=await browser.newPage();
 const errors=[];
 page.on('pageerror',error=>errors.push(error.message));
 await page.goto('http://127.0.0.1:5173');
 await page.getByLabel('Username',{exact:true}).fill('operator');
 await page.getByLabel('Password',{exact:true}).fill(process.env.AUDITORY_CHECK_PASSWORD);
 await page.getByRole('button',{name:/Sign in/}).click();
 const history=page.getByRole('region',{name:'Session history',exact:true});
 await history.getByLabel('Saved session',{exact:true}).selectOption(native.fits[0].session_id);
 await history.getByRole('button',{name:'View '+native.fits[0].fit_id.slice(0,8),exact:true}).click();
 await history.getByText('Saved fit · succeeded',{exact:true}).waitFor();
 assert.equal(await history.getByLabel('Saved result',{exact:true}).locator('tbody tr').count(),18);
 await page.screenshot({path:root+'/issue11-deployed-native.png',fullPage:true});
 assert.deepEqual(errors,[]);
 console.log('PASS: packaged backend and separately served production frontend show 18 real mVocs correlations using served runtime API config.');
} finally { await browser.close(); }
