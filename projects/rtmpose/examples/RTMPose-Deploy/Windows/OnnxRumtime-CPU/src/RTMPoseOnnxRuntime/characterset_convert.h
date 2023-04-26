#ifndef _CHARACTERSET_CONVERT_H_
#define _CHARACTERSET_CONVERT_H_

#include <string>

#if defined(WIN32) || defined(_WIN32) || defined(__WIN32__) || defined(__NT__)

#include <Windows.h>

#elif defined(linux) || defined(__linux)

#include <iconv.h>
#include <malloc.h>

#endif

namespace stubbornhuang
{
	class CharactersetConvert
	{
	public:
#if defined(WIN32) || defined(_WIN32) || defined(__WIN32__) || defined(__NT__)
		static std::wstring string_to_wstring(const std::string& str)
		{
			std::wstring result;
			int len = MultiByteToWideChar(CP_ACP, 0, str.c_str(), -1, NULL, 0);
			wchar_t* wstr = new wchar_t[len + 1];
			memset(wstr, 0, len + 1);
			MultiByteToWideChar(CP_ACP, 0, str.c_str(), -1, wstr, len);
			wstr[len] = '\0';
			result.append(wstr);
			delete[] wstr;
			return result;
		}

		static std::string gbk_to_utf8(const std::string& gbk_str)
		{
			int len = MultiByteToWideChar(CP_ACP, 0, gbk_str.c_str(), -1, NULL, 0);
			wchar_t* wstr = new wchar_t[len + 1];
			memset(wstr, 0, len + 1);
			MultiByteToWideChar(CP_ACP, 0, gbk_str.c_str(), -1, wstr, len);
			len = WideCharToMultiByte(CP_UTF8, 0, wstr, -1, NULL, 0, NULL, NULL);
			char* str = new char[len + 1];
			memset(str, 0, len + 1);
			WideCharToMultiByte(CP_UTF8, 0, wstr, -1, str, len, NULL, NULL);
			std::string strTemp = str;
			if (wstr) delete[] wstr;
			if (str) delete[] str;
			return strTemp;
		}

		static std::string utf8_to_gbk(const std::string& utf8_str)
		{
			int len = MultiByteToWideChar(CP_UTF8, 0, utf8_str.c_str(), -1, NULL, 0);
			wchar_t* wszGBK = new wchar_t[len + 1];
			memset(wszGBK, 0, len * 2 + 2);
			MultiByteToWideChar(CP_UTF8, 0, utf8_str.c_str(), -1, wszGBK, len);
			len = WideCharToMultiByte(CP_ACP, 0, wszGBK, -1, NULL, 0, NULL, NULL);
			char* szGBK = new char[len + 1];
			memset(szGBK, 0, len + 1);
			WideCharToMultiByte(CP_ACP, 0, wszGBK, -1, szGBK, len, NULL, NULL);
			std::string strTemp(szGBK);
			if (wszGBK) delete[] wszGBK;
			if (szGBK) delete[] szGBK;
			return strTemp;
		}

#elif defined(linux) || defined(__linux)
		static int code_convert(
			const char* from_charset,
			const char* to_charset,
			char* inbuf, size_t inlen,
			char* outbuf, size_t outlen
		) {
			iconv_t cd;
			char** pin = &inbuf;
			char** pout = &outbuf;

			cd = iconv_open(to_charset, from_charset);
			if (cd == 0)
				return -1;

			memset(outbuf, 0, outlen);

			if ((int)iconv(cd, pin, &inlen, pout, &outlen) == -1)
			{
				iconv_close(cd);
				return -1;
			}
			iconv_close(cd);
			*pout = '\0';

			return 0;
		}

		static int u2g(char* inbuf, size_t inlen, char* outbuf, size_t outlen) {
			return code_convert("utf-8", "gb2312", inbuf, inlen, outbuf, outlen);
		}

		static int g2u(char* inbuf, size_t inlen, char* outbuf, size_t outlen) {
			return code_convert("gb2312", "utf-8", inbuf, inlen, outbuf, outlen);
		}


		static std::string gbk_to_utf8(const std::string& gbk_str)
		{
			int length = gbk_str.size() * 2 + 1;

			char* temp = (char*)malloc(sizeof(char) * length);

			if (g2u((char*)gbk_str.c_str(), gbk_str.size(), temp, length) >= 0)
			{
				std::string str_result;
				str_result.append(temp);
				free(temp);
				return str_result;
	}
			else
			{
				free(temp);
				return "";
			}
		}

		static std::string utf8_to_gbk(const std::string& utf8_str)
		{
			int length = strlen(utf8_str);

			char* temp = (char*)malloc(sizeof(char) * length);

			if (u2g((char*)utf8_str, length, temp, length) >= 0)
			{
				std::string str_result;
				str_result.append(temp);
				free(temp);

				return str_result;
			}
			else
			{
				free(temp);
				return "";
			}
		}

#endif

	};
}

#endif // !_CHARACTERSET_CONVERT_H_
