#!/usr/bin/env python3
"""
Lambda Instance Termination Reminder System
Prevents costly GPU billing by ensuring instances are terminated when work is complete
"""

import time
import json
import os
import subprocess
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Any, Optional

class LambdaTerminationReminder:
    """Manages Lambda instance termination reminders and auto-shutdown."""
    
    def __init__(self):
        self.reminder_file = Path.home() / ".lambda_reminder.json"
        self.alert_intervals = [1, 2, 4, 8, 12, 24]  # Hours
        
    def register_instance(self, instance_id: str, instance_ip: str, 
                         estimated_duration_hours: float, cost_per_hour: float = 1.50):
        """Register a Lambda instance for termination monitoring."""
        
        start_time = datetime.now()
        estimated_completion = start_time + timedelta(hours=estimated_duration_hours)
        
        instance_info = {
            'instance_id': instance_id,
            'instance_ip': instance_ip,
            'start_time': start_time.isoformat(),
            'estimated_duration_hours': estimated_duration_hours,
            'estimated_completion': estimated_completion.isoformat(),
            'cost_per_hour': cost_per_hour,
            'estimated_total_cost': estimated_duration_hours * cost_per_hour,
            'alerts_sent': [],
            'status': 'running',
            'purpose': 'GPU Autotuning V2',
            'registered_by': os.getenv('USER', 'unknown')
        }
        
        # Save to reminder file
        self.save_instance_info(instance_info)
        
        print(f"🔔 Lambda Instance Registered for Termination Reminders")
        print(f"   Instance: {instance_id}")
        print(f"   IP: {instance_ip}")
        print(f"   Estimated duration: {estimated_duration_hours:.1f} hours")
        print(f"   Estimated cost: ${instance_info['estimated_total_cost']:.2f}")
        print(f"   Estimated completion: {estimated_completion.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"   Hourly rate: ${cost_per_hour:.2f}")
        
        return instance_info
    
    def save_instance_info(self, instance_info: Dict[str, Any]):
        """Save instance information to reminder file."""
        data = {'instances': []}
        
        # Load existing data if present
        if self.reminder_file.exists():
            try:
                with open(self.reminder_file, 'r') as f:
                    data = json.load(f)
            except:
                pass
        
        # Add new instance
        data['instances'].append(instance_info)
        
        # Save updated data
        with open(self.reminder_file, 'w') as f:
            json.dump(data, f, indent=2)
    
    def check_running_instances(self) -> Dict[str, Any]:
        """Check status of all registered instances and send alerts if needed."""
        if not self.reminder_file.exists():
            return {'instances': [], 'alerts': []}
        
        try:
            with open(self.reminder_file, 'r') as f:
                data = json.load(f)
        except:
            return {'instances': [], 'alerts': []}
        
        current_time = datetime.now()
        alerts = []
        
        for instance in data.get('instances', []):
            if instance.get('status') != 'running':
                continue
                
            start_time = datetime.fromisoformat(instance['start_time'])
            running_hours = (current_time - start_time).total_seconds() / 3600
            current_cost = running_hours * instance['cost_per_hour']
            
            # Check if we should send alerts
            for alert_hour in self.alert_intervals:
                if (running_hours >= alert_hour and 
                    alert_hour not in instance.get('alerts_sent', [])):
                    
                    alert = self.generate_alert(instance, running_hours, current_cost)
                    alerts.append(alert)
                    instance['alerts_sent'].append(alert_hour)
        
        # Save updated data
        with open(self.reminder_file, 'w') as f:
            json.dump(data, f, indent=2)
        
        return {'instances': data.get('instances', []), 'alerts': alerts}
    
    def generate_alert(self, instance: Dict[str, Any], running_hours: float, 
                      current_cost: float) -> Dict[str, Any]:
        """Generate termination alert for instance."""
        estimated_hours = instance['estimated_duration_hours']
        
        alert = {
            'timestamp': datetime.now().isoformat(),
            'instance_id': instance['instance_id'],
            'instance_ip': instance['instance_ip'],
            'running_hours': running_hours,
            'estimated_hours': estimated_hours,
            'current_cost': current_cost,
            'estimated_cost': instance['estimated_total_cost'],
            'hourly_rate': instance['cost_per_hour'],
            'purpose': instance.get('purpose', 'Unknown'),
            'alert_type': self.get_alert_type(running_hours, estimated_hours)
        }
        
        self.print_alert(alert)
        self.send_system_notification(alert)
        
        return alert
    
    def get_alert_type(self, running_hours: float, estimated_hours: float) -> str:
        """Determine alert type based on running time."""
        if running_hours > estimated_hours * 2:
            return 'CRITICAL_OVERRUN'
        elif running_hours > estimated_hours:
            return 'OVERRUN'
        elif running_hours > estimated_hours * 0.8:
            return 'COMPLETION_SOON'
        else:
            return 'PROGRESS_UPDATE'
    
    def print_alert(self, alert: Dict[str, Any]):
        """Print termination alert to console."""
        alert_type = alert['alert_type']
        
        if alert_type == 'CRITICAL_OVERRUN':
            icon = "🚨"
            message = "CRITICAL: Instance running much longer than expected!"
        elif alert_type == 'OVERRUN':
            icon = "⚠️"
            message = "WARNING: Instance exceeded estimated duration!"
        elif alert_type == 'COMPLETION_SOON':
            icon = "🔔"
            message = "INFO: Instance should complete soon"
        else:
            icon = "📊"
            message = "UPDATE: Instance progress report"
        
        print(f"\n{icon} LAMBDA TERMINATION ALERT {icon}")
        print("=" * 50)
        print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Message: {message}")
        print(f"Instance: {alert['instance_id']}")
        print(f"IP: {alert['instance_ip']}")
        print(f"Purpose: {alert['purpose']}")
        print(f"Running: {alert['running_hours']:.1f}h / {alert['estimated_hours']:.1f}h estimated")
        print(f"Cost: ${alert['current_cost']:.2f} / ${alert['estimated_cost']:.2f} estimated")
        print(f"Rate: ${alert['hourly_rate']:.2f}/hour")
        
        if alert_type in ['OVERRUN', 'CRITICAL_OVERRUN']:
            print(f"\n💰 COST OVERRUN: ${alert['current_cost'] - alert['estimated_cost']:.2f}")
            print("🔧 ACTION REQUIRED:")
            print("   1. Check if work is complete")
            print("   2. SSH to instance and verify status")
            print("   3. TERMINATE instance if work is done")
            print(f"   4. SSH command: ssh ubuntu@{alert['instance_ip']}")
            print("   5. Termination: Use Lambda Labs dashboard")
        
        print("=" * 50)
    
    def send_system_notification(self, alert: Dict[str, Any]):
        """Send system notification (macOS/Linux)."""
        try:
            title = f"Lambda GPU Alert - {alert['alert_type']}"
            message = (f"Instance {alert['instance_id']} has been running "
                      f"{alert['running_hours']:.1f}h. Current cost: ${alert['current_cost']:.2f}")
            
            # Try macOS notification
            subprocess.run([
                'osascript', '-e', 
                f'display notification "{message}" with title "{title}"'
            ], capture_output=True)
        except:
            # Try Linux notification
            try:
                subprocess.run([
                    'notify-send', title, message
                ], capture_output=True)
            except:
                pass  # No notification system available
    
    def mark_instance_terminated(self, instance_id: str):
        """Mark instance as terminated to stop alerts."""
        if not self.reminder_file.exists():
            return
        
        try:
            with open(self.reminder_file, 'r') as f:
                data = json.load(f)
        except:
            return
        
        for instance in data.get('instances', []):
            if instance['instance_id'] == instance_id:
                instance['status'] = 'terminated'
                instance['termination_time'] = datetime.now().isoformat()
                
                start_time = datetime.fromisoformat(instance['start_time'])
                total_hours = (datetime.now() - start_time).total_seconds() / 3600
                final_cost = total_hours * instance['cost_per_hour']
                
                instance['final_runtime_hours'] = total_hours
                instance['final_cost'] = final_cost
                
                print(f"✅ Instance {instance_id} marked as terminated")
                print(f"   Total runtime: {total_hours:.1f} hours")
                print(f"   Final cost: ${final_cost:.2f}")
                break
        
        with open(self.reminder_file, 'w') as f:
            json.dump(data, f, indent=2)
    
    def get_cost_summary(self) -> Dict[str, Any]:
        """Get cost summary for all instances."""
        if not self.reminder_file.exists():
            return {'total_cost': 0, 'active_cost': 0, 'instances': []}
        
        try:
            with open(self.reminder_file, 'r') as f:
                data = json.load(f)
        except:
            return {'total_cost': 0, 'active_cost': 0, 'instances': []}
        
        total_cost = 0
        active_cost = 0
        current_time = datetime.now()
        
        for instance in data.get('instances', []):
            if instance.get('status') == 'terminated':
                total_cost += instance.get('final_cost', 0)
            else:
                start_time = datetime.fromisoformat(instance['start_time'])
                running_hours = (current_time - start_time).total_seconds() / 3600
                cost = running_hours * instance['cost_per_hour']
                total_cost += cost
                active_cost += cost
        
        return {
            'total_cost': total_cost,
            'active_cost': active_cost,
            'instances': data.get('instances', [])
        }

def main():
    """Main function for command line usage."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Lambda Instance Termination Reminder')
    parser.add_argument('--register', help='Register instance (format: instance_id,ip,hours)')
    parser.add_argument('--check', action='store_true', help='Check running instances')
    parser.add_argument('--terminate', help='Mark instance as terminated')
    parser.add_argument('--cost-summary', action='store_true', help='Show cost summary')
    parser.add_argument('--watch', action='store_true', help='Watch mode (continuous monitoring)')
    
    args = parser.parse_args()
    reminder = LambdaTerminationReminder()
    
    if args.register:
        parts = args.register.split(',')
        if len(parts) != 3:
            print("❌ Register format: instance_id,ip,hours")
            return
        
        instance_id, ip, hours = parts
        reminder.register_instance(instance_id.strip(), ip.strip(), float(hours))
    
    elif args.check:
        result = reminder.check_running_instances()
        if result['alerts']:
            for alert in result['alerts']:
                print(f"Alert sent for {alert['instance_id']}")
        else:
            print("✅ No alerts needed")
    
    elif args.terminate:
        reminder.mark_instance_terminated(args.terminate)
    
    elif args.cost_summary:
        summary = reminder.get_cost_summary()
        print(f"💰 Lambda GPU Cost Summary")
        print(f"   Total cost: ${summary['total_cost']:.2f}")
        print(f"   Active cost: ${summary['active_cost']:.2f}")
        print(f"   Total instances: {len(summary['instances'])}")
    
    elif args.watch:
        print("👀 Lambda Termination Watch Mode (Ctrl+C to stop)")
        try:
            while True:
                reminder.check_running_instances()
                time.sleep(300)  # Check every 5 minutes
        except KeyboardInterrupt:
            print("\n✅ Watch mode stopped")
    
    else:
        parser.print_help()

if __name__ == "__main__":
    main()