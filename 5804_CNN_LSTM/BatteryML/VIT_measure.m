clc;
clear all;
close all;


load B0005.mat
FTime = B0005.cycle(1).data.Time/60;
FreshCell_V = B0005.cycle(1).data.Voltage_measured;
FreshCell_I = B0005.cycle(1).data.Current_measured;
FreshCell_T = B0005.cycle(1).data.Temperature_measured;

ATime = B0005.cycle(200).data.Time/60;
AgedCell_V = B0005.cycle(200).data.Voltage_measured;
AgedCell_I = B0005.cycle(200).data.Current_measured;
AgedCell_T = B0005.cycle(200).data.Temperature_measured;

figure(1);
subplot(311);
plot(FTime, FreshCell_V,'b','linewidth', 2) 
hold on;
plot(ATime, AgedCell_V, 'r--','linewidth', 2)
hold off; 
legend('Fresh Cell(1st Cycle)', 'Aged Cell(200th Cycle'), ylabel('Voltage(V)');
ylim([3.5 4.5]); 
xlim([0 140]);
grid on

subplot(312);
plot(FTime, FreshCell_I,'b', 'linewidth', 2)
hold on;
plot(ATime, AgedCell_I, 'r--', 'linewidth', 2)
hold off;
ylabel('Current(A)');
ylim([0 2]); 
xlim([0 140]),
grid on

subplot(313);
plot(FTime, FreshCell_T,'b', 'linewidth', 2)
hold on;
plot(ATime, AgedCell_T, 'r--', 'linewidth', 2)
hold off; 
ylabel('Temperature(^oC)');
ylim([22 30]); 
grid on;
xlabel Time(Minute);
xlim([0 140]);
