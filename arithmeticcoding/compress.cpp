#include <iostream>
#include <map>
#include <string>
#include <cstring>
#include<math.h>
using namespace std;
const int N = 256;
int len;//长度
string str;
int char_num[N];//统计字符
struct node {//字符及概率区间
	char c;
	double l, r;
}ch[N];
struct spcode {
	double low=0.0, high=1.0, delta;//区间左右端及大小
	spcode() = default;
	spcode(double a, double b) :low(a), high(b), delta(b-a) {}
};
map<char, spcode> mp;
void create() {
	memset(char_num, 0, sizeof(char_num));
	printf("输入字符串个数n\n");
	cin >> len;
	printf("输入字符串：\n");
	cin >> str;
	for (auto i = 0; i < str.length(); i++)//统计概率
		char_num[str[i]]++;
	double last = 0.0;
	for (int i = 0; i <N; i++) {
		if (char_num[i]) {
			ch[i].c = i;
			ch[i].l = last;
			ch[i].r = last + double(char_num[i]) / len;
			last = ch[i].r;
		}
	}
	for (auto i = 0; i < str.length(); i++) {
		if (char_num[str[i]]) {
			mp.insert(make_pair(str[i], spcode(ch[str[i]].l,ch[str[i]].r)));
		}
	}
}
string encode(double &db,string str){
	double low = 0.0;
	double high = 1.0;
	for (auto it = str.begin(); it != str.end(); it++) {
		double delta = high - low;
		high = low + delta * mp[*it].high;
		low = low + delta * mp[*it].low;
	}
	//寻找最短二进制
	string anstr = "";
	double ans = 0.0;
	int cnt = 1;
	while (ans < low) {
		ans += pow(0.5, cnt);
		anstr += '1';
		if (ans >= high) {
			ans -= pow(0.5, cnt);
			anstr[cnt-1] = '0';
		}
		cnt++;
	}
	db = ans;
	return anstr;
}
string decode(double value) {
	double low, high;                   //编码区间
	double prelow = 0.0, prehigh = 1.0;//记录前一次
	string ans = "";
	int cur = 0;
	while (true) {
		low = prelow;
		high = prehigh;
		for (auto it = mp.begin(); it != mp.end(); it++) {
			double delta = high - low;
			high = low + delta * it->second.high;
			low = low + delta * it->second.low;
			if (value>=low && value<high) {
				prelow = low;
				prehigh = high;
				ans += (it->first);
				cur++;
				break;
			}else
			{
				low = prelow;
				high = prehigh;
			}
		}
		if (cur == len)break;
	}
	return ans;
}
int main() {
	spcode sp;
	create();
	cout << "\n";
	/*test map
	for (auto it = mp.begin(); it != mp.end(); it++) {
		cout << it->first <<  " (";
		cout << it->second.low << " ";
		cout << it->second.high << ") ";
		cout << it->second.delta << "\n";
	}
	*/
	double db;
	string anstr = encode(db, str);
	cout << "转化为小数的最短十进制表示为:\n" << db << endl;
	cout << "\n";
	cout << "最短二进制码为:\n" << anstr << endl;
	cout << "\n";
	string destr=decode(db);
	cout <<"解码后的字符串为: \n"<< destr << endl;

	return 0;
}
