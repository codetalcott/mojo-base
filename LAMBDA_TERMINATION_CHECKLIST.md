# 🚨 LAMBDA INSTANCE TERMINATION CHECKLIST

## ⚠️ CRITICAL BILLING WARNING
**Lambda Cloud charges CONTINUOUSLY while GPU instances are running - even when idle!**
- **A10 GPU Rate**: ~$1.50/hour
- **Daily Cost**: $36/day if left running
- **Monthly Cost**: $1,080/month if forgotten

## 📋 Pre-Launch Checklist

### Before Starting Autotuning:
- [ ] Set phone alarms for 2, 4, 6, and 8 hours from now
- [ ] Add calendar reminders to check instance status
- [ ] Note Lambda Labs dashboard login credentials
- [ ] Bookmark Lambda Labs instance management page
- [ ] Verify termination monitoring is active

### During Autotuning:
- [ ] Monitor termination reminder alerts
- [ ] Check progress every 2 hours
- [ ] Verify results are being saved correctly

## 🚨 Post-Completion Checklist (CRITICAL!)

### Immediately After Autotuning Completes:
- [ ] **Download all results to local machine**
- [ ] **Verify autotuning_results/*.json files are saved**
- [ ] **Check for any error logs that need investigation**
- [ ] **Confirm no additional data needs to be retrieved**

### Termination Process:
- [ ] **Go to Lambda Labs dashboard**
- [ ] **Navigate to "Instances" section**
- [ ] **Find your A10 GPU instance**
- [ ] **Click "TERMINATE" (not "Stop"!)**
- [ ] **Confirm termination in popup**
- [ ] **Wait for "Terminated" status confirmation**

### Verification:
- [ ] **Instance shows "Terminated" status**
- [ ] **No active billing for the instance**
- [ ] **Mark instance as terminated in reminder system:**
  ```bash
  python3 scripts/lambda_termination_reminder.py --terminate INSTANCE_ID
  ```

## 🔔 Termination Reminder System

### Automatic Monitoring:
The system automatically sends alerts at:
- 1 hour: Progress update
- 2 hours: Status check
- 4 hours: Should be complete - consider termination
- 8 hours: Critical overrun - terminate unless emergency
- 12+ hours: Emergency - terminate immediately

### Manual Commands:
```bash
# Check current status
python3 scripts/lambda_termination_reminder.py --check

# Get cost summary
python3 scripts/lambda_termination_reminder.py --cost-summary

# Mark as terminated
python3 scripts/lambda_termination_reminder.py --terminate INSTANCE_ID

# Continuous monitoring
python3 scripts/lambda_termination_reminder.py --watch
```

## 💡 Cost Management Tips

### Expected Costs:
- **Normal autotuning (2-4 hours)**: $3-6
- **Extended run (6 hours)**: $9
- **Overnight mistake (12 hours)**: $18
- **Full day mistake (24 hours)**: $36
- **Week-long mistake**: $252

### If You Forget to Terminate:
1. **Terminate immediately** - every minute costs money
2. **Check billing dashboard** for current charges
3. **Contact Lambda support** if charges seem incorrect
4. **Learn from mistake** - set better reminders next time

## 📱 Phone Reminder Template

### Calendar Event Template:
```
Title: "🚨 CHECK LAMBDA GPU INSTANCE"
Time: [2/4/6/8 hours after start]
Description: 
- Check if autotuning is complete
- Download results if done
- TERMINATE INSTANCE to stop billing
- Lambda charges $1.50/hour continuously!
Location: Lambda Labs Dashboard
```

### Phone Alarm Template:
- **2 hours**: "Lambda GPU - Check Progress"
- **4 hours**: "Lambda GPU - Should be done - TERMINATE"
- **6 hours**: "Lambda GPU - CRITICAL - Why still running?"
- **8 hours**: "Lambda GPU - EMERGENCY - TERMINATE NOW!"

## 🎯 Success Metrics

### Autotuning V2 Success:
- [ ] Real GPU performance data collected
- [ ] 100+ configurations tested
- [ ] Production-scale corpus used (50K+ vectors)
- [ ] Results saved locally
- [ ] V1 vs V2 comparison completed
- [ ] **Instance terminated within 6 hours**

### Cost Success:
- [ ] **Total cost under $10**
- [ ] **No surprise billing charges**
- [ ] **Instance properly terminated**
- [ ] **No ongoing charges after completion**

## 🚨 Emergency Contacts

If you can't access Lambda Labs dashboard:
- **Lambda Labs Support**: Available through dashboard
- **Email**: Check Lambda Labs documentation
- **Alternative**: Contact teammate with Lambda access

## 💾 Results Backup

Before terminating, ensure these files are saved locally:
```
autotuning_results/autotune_v2_YYYYMMDD_HHMMSS_results.json
autotuning_results/autotune_v2_YYYYMMDD_HHMMSS_optimized_kernel.mojo
docs/AUTOTUNING_V2_*.md
Any log files or error reports
```

---

## 🎉 Final Checklist

**Before clicking "Terminate":**
- [ ] All results downloaded ✅
- [ ] No critical errors needing investigation ✅
- [ ] All needed data retrieved ✅
- [ ] Ready to terminate instance ✅

**After termination:**
- [ ] Instance shows "Terminated" status ✅
- [ ] No active billing ✅
- [ ] Termination recorded in reminder system ✅
- [ ] Success! 🎉

---

**REMEMBER: Every minute the instance runs costs money. TERMINATE IMMEDIATELY when work is complete!**